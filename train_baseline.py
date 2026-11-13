"""
train_baseline.py - Deliverable 1: train the FP32 ResNet-18 baseline on CIFAR-10.

    python train_baseline.py --epochs 20

Output: checkpoints/resnet18_fp32.pth  +  logs/baseline_train_log.csv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from common import (CKPT_DIR, LOG_DIR, Cifar10Data, banner, build_resnet18, count_parameters,
                    ensure_dirs, evaluate, get_device, save_checkpoint, set_seed,
                    train_one_epoch)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1, help="peak learning rate (one-cycle)")
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(CKPT_DIR / "resnet18_fp32.pth"))
    args = ap.parse_args()

    ensure_dirs()
    set_seed(args.seed)
    device = get_device(args.device)
    data = Cifar10Data()
    model = build_resnet18().to(device)
    banner(f"Baseline FP32 ResNet-18 | device={device} | params={count_parameters(model):,}")

    steps_per_epoch = len(data.y_train) // args.batch_size
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.2)
    log_path = LOG_DIR / "baseline_train_log.csv"
    best_acc, t_start = 0.0, time.time()
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_acc", "lr", "epoch_time_s"])
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss, tr_acc = train_one_epoch(model, data, opt, device, sched, args.batch_size)
            te_acc = evaluate(model, data, device)
            lr = opt.param_groups[0]["lr"]
            writer.writerow([epoch, f"{loss:.4f}", f"{tr_acc:.2f}", f"{te_acc:.2f}", f"{lr:.5f}",
                             f"{time.time() - t0:.1f}"])
            f.flush()
            print(f"epoch {epoch:3d}/{args.epochs} | loss {loss:.4f} | train {tr_acc:6.2f}% | "
                  f"test {te_acc:6.2f}% | lr {lr:.4f} | {time.time() - t0:5.1f}s", flush=True)
            best_acc = max(best_acc, te_acc)

    # The one-cycle schedule ends at (near) zero LR, so the final epoch is the model we ship.
    final_acc = evaluate(model, data, device)
    save_checkpoint(Path(args.out), model, test_acc=final_acc,
                    epochs=args.epochs, seed=args.seed)
    print(f"\nFinal FP32 test accuracy: {final_acc:.2f}%  (best epoch {best_acc:.2f}%)")
    print(f"Total training time: {(time.time() - t_start) / 60:.1f} min")
    print(f"Saved checkpoint -> {args.out}")
    print(f"Training log     -> {log_path}")


if __name__ == "__main__":
    main()
