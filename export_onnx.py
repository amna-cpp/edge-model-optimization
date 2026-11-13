"""
export_onnx.py - Deliverable 8 (part 1): export all five models to ONNX for CPU deployment.

  FP32 / unstructured / structured : direct torch.onnx.export of the float network.
  PTQ-INT8 and QAT-INT8            : PyTorch eager-mode INT8 operators (packed QNNPACK weights) have no
                                     ONNX counterpart, so the converted INT8 model is exported through a
                                     QDQ re-expression: the exact INT8 weights, per-channel scales,
                                     activation scales and zero-points are read out of the converted
                                     PyTorch model and written as QuantizeLinear / DequantizeLinear
                                     nodes around float Conv / Gemm / Add / ReLU.  ONNX-Runtime fuses
                                     those patterns into true INT8 kernels (QLinearConv ...), so the ONNX
                                     INT8 models carry the very same quantisation parameters as the
                                     PyTorch INT8 checkpoints (PTQ calibration / QAT-learned scales).

    python export_onnx.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import convert

from common import (CKPT_DIR, ONNX_DIR, Cifar10Data, banner, build_resnet18, ensure_dirs, evaluate,
                    load_float_model, select_quant_backend, set_seed)
from inference import MODELS
from quant_utils import build_ptq_prepared, build_qat_prepared

warnings.filterwarnings("ignore")
OPSET = 13


# --------------------------------------------------------------------------- #
# float models
# --------------------------------------------------------------------------- #
def export_onnx(model: nn.Module, path: Path) -> None:
    torch.onnx.export(model.cpu().eval(), torch.randn(1, 3, 32, 32), str(path), input_names=["input"],
                      output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=OPSET, do_constant_folding=True, dynamo=False)


# --------------------------------------------------------------------------- #
# INT8 models -> QDQ graph
# --------------------------------------------------------------------------- #
def load_int8_from_state(state_path: Path, qat: bool) -> nn.Module:
    """Rebuild the converted INT8 module tree and load the saved quantised state_dict into it."""
    backend = select_quant_backend()
    prepared = build_qat_prepared(None, backend) if qat else build_ptq_prepared(None, backend)
    prepared.eval()
    prepared(torch.randn(8, 3, 32, 32))               # let observers see data so convert() works
    qmodel = convert(prepared, inplace=False)
    qmodel.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=False)["state_dict"])
    return qmodel.eval()


def fold_int8_weights(raw_path: Path, out_path: Path) -> None:
    """Constant-fold QuantizeLinear(weight) into real INT8 initialisers (ORT basic, standard-ONNX-only
    optimisations) -> ~4x smaller file; outputs are bit-identical to the unfolded graph."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(out_path)
    ort.InferenceSession(str(raw_path), opts, providers=["CPUExecutionProvider"])


def _act_q(x: torch.Tensor, scale, zp) -> torch.Tensor:
    """Per-tensor UINT8 fake-quant  ->  QuantizeLinear + DequantizeLinear in ONNX."""
    return torch.fake_quantize_per_tensor_affine(x, float(scale), int(zp), 0, 255)


class QDQLayer(nn.Module):
    """A quantised Conv2d / Linear (optionally + ReLU) re-expressed with QDQ nodes."""

    def __init__(self, qlayer: nn.Module, relu: bool, is_conv: bool):
        super().__init__()
        w = qlayer.weight()
        self.is_conv, self.relu = is_conv, relu
        self.register_buffer("w_float", w.dequantize())
        self.register_buffer("bias", qlayer.bias().detach().clone())
        self.per_channel = w.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric)
        if self.per_channel:
            self.register_buffer("w_scale", w.q_per_channel_scales().float())
            self.register_buffer("w_zp", w.q_per_channel_zero_points().to(torch.int32))
            self.axis = w.q_per_channel_axis()
        else:
            self.w_scale, self.w_zp = float(w.q_scale()), int(w.q_zero_point())
        self.out_scale, self.out_zp = float(qlayer.scale), int(qlayer.zero_point)
        if is_conv:
            self.stride, self.padding = tuple(qlayer.stride), tuple(qlayer.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.per_channel:
            w = torch.fake_quantize_per_channel_affine(self.w_float, self.w_scale, self.w_zp, self.axis, -128, 127)
        else:
            w = torch.fake_quantize_per_tensor_affine(self.w_float, self.w_scale, self.w_zp, -128, 127)
        y = F.conv2d(x, w, self.bias, self.stride, self.padding) if self.is_conv else F.linear(x, w, self.bias)
        if self.relu:
            y = F.relu(y)
        return _act_q(y, self.out_scale, self.out_zp)


class QDQBlock(nn.Module):
    def __init__(self, qblock: nn.Module):
        super().__init__()
        self.conv1 = QDQLayer(qblock.conv1, relu=True, is_conv=True)
        self.conv2 = QDQLayer(qblock.conv2, relu=False, is_conv=True)
        self.shortcut = QDQLayer(qblock.shortcut[0], relu=False, is_conv=True) if len(qblock.shortcut) > 0 else None
        self.add_scale, self.add_zp = float(qblock.skip_add.scale), int(qblock.skip_add.zero_point)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        sc = x if self.shortcut is None else self.shortcut(x)
        return _act_q(F.relu(out + sc), self.add_scale, self.add_zp)


class QDQResNet18(nn.Module):
    """Float-op / QDQ-node twin of a converted INT8 ResNet-18."""

    def __init__(self, qmodel: nn.Module):
        super().__init__()
        self.in_scale, self.in_zp = float(qmodel.quant.scale), int(qmodel.quant.zero_point)
        self.conv1 = QDQLayer(qmodel.conv1, relu=True, is_conv=True)
        self.blocks = nn.ModuleList([QDQBlock(b) for s in (qmodel.layer1, qmodel.layer2, qmodel.layer3, qmodel.layer4)
                                     for b in s])
        last = qmodel.layer4[1].skip_add
        self.pool_scale, self.pool_zp = float(last.scale), int(last.zero_point)
        self.fc = QDQLayer(qmodel.fc, relu=False, is_conv=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _act_q(x, self.in_scale, self.in_zp)
        x = self.conv1(x)
        for b in self.blocks:
            x = b(x)
        x = _act_q(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1), self.pool_scale, self.pool_zp)
        return self.fc(x)


@torch.no_grad()
def check_qdq_fidelity(qmodel: nn.Module, qdq: nn.Module, data: Cifar10Data) -> tuple[float, float]:
    """(argmax agreement %, mean |logit diff|) between the PyTorch INT8 model and its QDQ twin."""
    x, _ = next(data.test_batches(500, torch.device("cpu")))
    a, b = qmodel(x), qdq(x)
    return 100.0 * (a.argmax(1) == b.argmax(1)).float().mean().item(), (a - b).abs().mean().item()


def main() -> None:
    ensure_dirs()
    set_seed()
    cpu = torch.device("cpu")

    banner("Export FP32 / pruned float models")
    for key, fname in [("fp32", "resnet18_fp32.onnx"), ("unstructured", "resnet18_unstructured_pruned.onnx"),
                       ("structured", "resnet18_structured_pruned.onnx")]:
        out = ONNX_DIR / fname
        export_onnx(load_float_model(MODELS[key][1], cpu), out)
        onnx.checker.check_model(onnx.load(str(out)))
        print(f"{key:13s} -> {out.name}  ({out.stat().st_size / 1e6:.2f} MB)  [onnx checker OK]")

    data = Cifar10Data()
    for key, state, qat in [("ptq", "resnet18_ptq_int8.pth", False), ("qat", "resnet18_qat_int8.pth", True)]:
        banner(f"Export {key.upper()}-INT8 (QDQ re-expression of the converted INT8 model)")
        qmodel = load_int8_from_state(CKPT_DIR / state, qat)
        acc_pt = evaluate(qmodel, data, cpu)
        qdq = QDQResNet18(qmodel).eval()
        agree, mad = check_qdq_fidelity(qmodel, qdq, data)
        print(f"PyTorch INT8 accuracy (rebuilt from state_dict): {acc_pt:.2f}%")
        print(f"QDQ twin vs INT8 model on 500 images: argmax agreement {agree:.1f}%, mean |logit diff| {mad:.4f}")
        out = ONNX_DIR / f"resnet18_{key}_int8.onnx"
        raw = ONNX_DIR / f"_raw_{key}.onnx"
        export_onnx(qdq, raw)
        fold_int8_weights(raw, out)
        raw.unlink()
        m = onnx.load(str(out))
        onnx.checker.check_model(m)
        ops = {}
        for n in m.graph.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        print(f"{key:13s} -> {out.name}  ({out.stat().st_size / 1e6:.2f} MB)  [onnx checker OK]")
        print(f"  QuantizeLinear: {ops.get('QuantizeLinear', 0)} | DequantizeLinear: {ops.get('DequantizeLinear', 0)} "
              f"| Conv: {ops.get('Conv', 0)} | Gemm/MatMul: {ops.get('Gemm', 0) + ops.get('MatMul', 0)}")


if __name__ == "__main__":
    main()
