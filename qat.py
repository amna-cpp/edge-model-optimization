"""
qat.py - Deliverable 3: Quantisation-Aware Training (QAT) of ResNet-18.

Pipeline
  1. initialise from the pretrained FP32 checkpoint
  2. fuse Conv+BN+ReLU (QAT variant, train mode)
  3. insert FakeQuantize modules (prepare_qat)
  4. fine-tune with fake quantisation in the forward pass
       - observers are frozen after `--freeze-observer-epoch`
       - BN statistics are frozen after `--freeze-bn-epoch`
  5. convert to a real INT8 model
  6. evaluate, save state_dict + TorchScript checkpoints
     (+ the fake-quant model, used for the ONNX QDQ export)

    python qat.py --epochs 3 --max-images 20000
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.ao.nn.intrinsic.qat as nniqat
from torch.ao.quantization import disable_observer

from common import (CKPT_DIR, LOG_DIR, Cifar10Data, banner, ensure_dirs, evaluate, file_size_mb,
                    get_device, load_float_model, select_quant_backend, set_seed,
                    train_one_epoch)
from quant_utils import build_qat_prepared, load_int8_torchscript, save_int8, to_int8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", default=str(CKPT_DIR / "resnet18_fp32.pth"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.005, help="initial LR (cosine decay)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-images", type=int, default=20000, help="random training images per epoch (CPU QAT is slow; ~12 min per full epoch)")
    ap.add_argument("--freeze-observer-epoch", type=int, default=3, help="disable observers from this epoch")
    ap.add_argument("--freeze-bn-epoch", type=int, default=2, help="freeze BN statistics from this epoch")
    ap.add_argument("--device", default="cpu", help="QAT device (cpu is the safe choice for fake-quant ops)")
    ap.add_argument("--out-state", default=str(CKPT_DIR / "resnet18_qat_int8.pth"))
    ap.add_argument("--out-ts", default=str(CKPT_DIR / "resnet18_qat_int8_ts.pt"))
    ap.add_argument("--out-fq", default=str(CKPT_DIR / "resnet18_qat_fakequant.pth"))
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    device = get_device(args.device)
    cpu = torch.device("cpu")
    data = Cifar10Data()

    banner("QAT step 1/6 - load pretrained FP32 model")
    fp32_acc = evaluate(load_float_model(Path(args.fp32), cpu), data, cpu)
    print(f"FP32 test accuracy: {fp32_acc:.2f}%")

    banner("QAT step 2-3/6 - fuse (train mode) + insert FakeQuantize modules")
    backend = select_quant_backend()
    model = build_qat_prepared(Path(args.fp32), backend)
    n_fq = sum(1 for m in model.modules() if "FakeQuantize" in type(m).__name__)
    print(f"backend: {backend} | fake-quant modules inserted: {n_fq} | training device: {device}")
    model.to(device)
    acc0 = evaluate(model, data, device)
    print(f"accuracy right after prepare_qat (uncalibrated observers): {acc0:.2f}%")

    banner("QAT step 4/6 - fine-tuning with fake quantisation")
    steps = (args.max_images or len(data.y_train)) // args.batch_size
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)
    log_path = LOG_DIR / "qat_train_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "fakequant_test_acc", "epoch_time_s"])
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            if epoch >= args.freeze_observer_epoch:
                model.apply(disable_observer)
            if epoch >= args.freeze_bn_epoch:
                model.apply(nniqat.freeze_bn_stats)
            loss, tr = train_one_epoch(model, data, opt, device, sched, args.batch_size, args.max_images)
            te = evaluate(model, data, device)
            w.writerow([epoch, f"{loss:.4f}", f"{tr:.2f}", f"{te:.2f}", f"{time.time() - t0:.1f}"])
            f.flush()
            print(f"epoch {epoch}/{args.epochs} | loss {loss:.4f} | train {tr:.2f}% | "
                  f"fake-quant test {te:.2f}% | {time.time() - t0:.0f}s", flush=True)

    banner("QAT step 5-6/6 - convert to INT8, evaluate, save")
    model.cpu().eval()
    torch.save({"state_dict": model.state_dict(), "backend": backend}, args.out_fq)
    int8 = to_int8(model)
    int8_acc = evaluate(int8, data, cpu)
    save_int8(int8, Path(args.out_state), Path(args.out_ts))
    print(f"INT8 (QAT) test accuracy : {int8_acc:.2f}%   (FP32 {fp32_acc:.2f}%, drop {fp32_acc - int8_acc:+.2f} pts)")
    print(f"state_dict checkpoint    : {args.out_state}  ({file_size_mb(Path(args.out_state)):.2f} MB)")
    print(f"TorchScript checkpoint   : {args.out_ts}  ({file_size_mb(Path(args.out_ts)):.2f} MB)")
    print(f"fake-quant checkpoint    : {args.out_fq}  (for ONNX QDQ export)")
    re_acc = evaluate(load_int8_torchscript(Path(args.out_ts)), data, cpu)
    print(f"reloaded TorchScript acc : {re_acc:.2f}%  ({'OK' if abs(re_acc - int8_acc) < 1e-6 else 'MISMATCH'})")


if __name__ == "__main__":
    main()
