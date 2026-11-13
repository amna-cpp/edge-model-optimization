"""
quant_utils.py - helpers shared by ptq.py, qat.py, inference.py and export_onnx.py.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.ao.quantization import (HistogramObserver, PerChannelMinMaxObserver, QConfig,
                                   convert, get_default_qat_qconfig, get_default_qconfig,
                                   prepare, prepare_qat)

from common import ResNet18, build_resnet18, load_float_model, select_quant_backend


def ptq_qconfig(backend: str, per_channel: bool = True) -> QConfig:
    """QConfig for post-training static quantisation.

    per_channel=True : INT8 weights with one scale per output channel (usually more accurate)
    per_channel=False: backend default (per-tensor weights)
    Activations always use a HistogramObserver (KL/L2-style range search) - more robust
    than plain min/max against activation outliers.
    """
    if not per_channel:
        return get_default_qconfig(backend)
    return QConfig(
        activation=HistogramObserver.with_args(dtype=torch.quint8, reduce_range=False),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
                                                  qscheme=torch.per_channel_symmetric),
    )


def build_ptq_prepared(fp32_ckpt: Path | None, backend: str, per_channel: bool = True) -> ResNet18:
    """FP32 checkpoint (or fresh architecture if None) -> fused -> observers inserted."""
    model = (load_float_model(fp32_ckpt, "cpu") if fp32_ckpt else build_resnet18()).eval()
    model.fuse_model(qat=False)                       # Conv+BN+ReLU -> ConvReLU2d / Conv2d
    model.qconfig = ptq_qconfig(backend, per_channel)
    return prepare(model, inplace=False)


def build_qat_prepared(fp32_ckpt: Path | None, backend: str) -> ResNet18:
    """FP32 checkpoint (or random init if None) -> fused (train mode) -> FakeQuant inserted."""
    model = load_float_model(fp32_ckpt, "cpu") if fp32_ckpt else build_resnet18()
    model.train()
    model.fuse_model(qat=True)                        # -> ConvBnReLU2d (QAT versions)
    model.qconfig = get_default_qat_qconfig(backend)
    return prepare_qat(model, inplace=False)


def to_int8(model: torch.nn.Module) -> torch.nn.Module:
    """Convert a calibrated / QAT-trained model to a real INT8 model (CPU)."""
    return convert(model.cpu().eval(), inplace=False)


def save_int8(model: torch.nn.Module, state_path: Path, ts_path: Path) -> None:
    """Persist an INT8 model twice:
       * state_dict     (.pth) - small, needs the Python class + same quant recipe to reload
       * TorchScript    (.pt)  - self-contained 'executable' checkpoint, loads with torch.jit.load
    """
    torch.save({"state_dict": model.state_dict(), "backend": torch.backends.quantized.engine},
               state_path)
    model.eval()
    try:
        scripted = torch.jit.script(model)
    except Exception as exc:                          # fall back to tracing
        print(f"  [warn] torch.jit.script failed ({type(exc).__name__}); tracing instead")
        scripted = torch.jit.trace(model, torch.randn(1, 3, 32, 32))
    torch.jit.save(scripted, str(ts_path))


def load_int8_torchscript(ts_path: Path) -> torch.nn.Module:
    select_quant_backend()                            # packed params are backend specific
    return torch.jit.load(str(ts_path), map_location="cpu").eval()
