"""
ptq.py - Deliverable 2: Post-Training Static Quantisation (PTQ) of the FP32 ResNet-18.

Pipeline
  1. load trained FP32 checkpoint
  2. fuse Conv+BN+ReLU
  3. select quantisation backend (fbgemm on x86 / qnnpack on ARM) and attach qconfig
  4. insert observers  (prepare)
  5. calibrate on a subset of the TRAINING set
  6. convert to INT8
  7. evaluate, save state_dict + TorchScript ("deployable") checkpoints

    python ptq.py --calib-images 2000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from calibration import calibrate
from common import (CKPT_DIR, Cifar10Data, banner, ensure_dirs, evaluate, file_size_mb,
                    load_float_model, select_quant_backend, set_seed)
from quant_utils import build_ptq_prepared, save_int8, to_int8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", default=str(CKPT_DIR / "resnet18_fp32.pth"))
    ap.add_argument("--calib-images", type=int, default=2000)
    ap.add_argument("--per-tensor", action="store_true",
                    help="per-tensor weight quantisation instead of per-channel")
    ap.add_argument("--out-state", default=str(CKPT_DIR / "resnet18_ptq_int8.pth"))
    ap.add_argument("--out-ts", default=str(CKPT_DIR / "resnet18_ptq_int8_ts.pt"))
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    cpu = torch.device("cpu")
    data = Cifar10Data()

    banner("PTQ step 1/6 - load FP32 model")
    fp32 = load_float_model(Path(args.fp32), cpu)
    fp32_acc = evaluate(fp32, data, cpu)
    print(f"FP32 test accuracy: {fp32_acc:.2f}%  ({file_size_mb(Path(args.fp32)):.2f} MB)")

    banner("PTQ step 2-3/6 - fuse layers + configure backend")
    backend = select_quant_backend()
    print(f"quantisation backend : {backend}   (supported: {torch.backends.quantized.supported_engines})")
    prepared = build_ptq_prepared(Path(args.fp32), backend, per_channel=not args.per_tensor)
    n_obs = sum(1 for m in prepared.modules() if "Observer" in type(m).__name__)
    print(f"weights              : {'per-tensor' if args.per_tensor else 'per-channel'} INT8 (symmetric)")
    print(f"activations          : UINT8 (histogram observer)")
    print(f"fused modules        : {sum(1 for m in prepared.modules() if 'ConvReLU' in type(m).__name__)} ConvReLU2d "
          f"+ conv-only (after BN folding)")
    print(f"observers inserted   : {n_obs}")

    banner("PTQ step 4-5/6 - calibration on training subset")
    used = calibrate(prepared, data, args.calib_images)
    print(f"calibrated with {used} training images")

    banner("PTQ step 6/6 - convert to INT8, evaluate, save")
    int8 = to_int8(prepared)
    int8_acc = evaluate(int8, data, cpu)
    save_int8(int8, Path(args.out_state), Path(args.out_ts))
    print(f"INT8 (PTQ) test accuracy : {int8_acc:.2f}%   (FP32 {fp32_acc:.2f}%, drop {fp32_acc - int8_acc:+.2f} pts)")
    print(f"state_dict checkpoint    : {args.out_state}  ({file_size_mb(Path(args.out_state)):.2f} MB)")
    print(f"TorchScript checkpoint   : {args.out_ts}  ({file_size_mb(Path(args.out_ts)):.2f} MB)")

    # sanity: the saved TorchScript file must reproduce the in-memory INT8 accuracy
    from quant_utils import load_int8_torchscript
    reloaded = load_int8_torchscript(Path(args.out_ts))
    re_acc = evaluate(reloaded, data, cpu)
    print(f"reloaded TorchScript acc : {re_acc:.2f}%  ({'OK' if abs(re_acc - int8_acc) < 1e-6 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
