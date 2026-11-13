"""
prune_unstructured.py - Deliverable 4: iterative magnitude-based UNSTRUCTURED pruning.

  * global L1-magnitude pruning over all Conv2d + Linear weights
  * several pruning iterations on a gradual (cubic) sparsity schedule,
    with fine-tuning between iterations to recover accuracy
  * weights are masked during the process (torch.nn.utils.prune adds weight_orig + weight_mask)
  * at the end the mask is baked in (prune.remove) -> permanent zeros in plain `weight` tensors

    python prune_unstructured.py --sparsity 0.9 --iterations 5
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from common import (CKPT_DIR, LOG_DIR, Cifar10Data, banner, count_nonzero_weights, ensure_dirs,
                    evaluate, file_size_mb, get_device, gzip_size_mb, load_float_model,
                    save_checkpoint, set_seed)
from prune_utils import finetune


def prunable_params(model: nn.Module):
    """All (module, 'weight') pairs of convolutional and fully connected layers."""
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def sparsity_of(model: nn.Module) -> float:
    """Fraction of exactly-zero weights (masked weights counted via weight_mask if present)."""
    zeros = total = 0
    for m, _ in prunable_params(model):
        w = m.weight                       # = weight_orig * weight_mask while a mask is attached
        zeros += int((w == 0).sum().item())
        total += w.numel()
    return zeros / total


def cubic_schedule(final_sparsity: float, iterations: int) -> list[float]:
    """Cumulative sparsity targets (Zhu & Gupta 2017): prune fast early, slowly at the end."""
    return [final_sparsity * (1 - (1 - t / iterations) ** 3) for t in range(1, iterations + 1)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", default=str(CKPT_DIR / "resnet18_fp32.pth"))
    ap.add_argument("--sparsity", type=float, default=0.90, help="final global sparsity")
    ap.add_argument("--iterations", type=int, default=5, help="number of pruning iterations")
    ap.add_argument("--ft-epochs-per-iter", type=int, default=1, help="fine-tune epochs after each prune step")
    ap.add_argument("--final-ft-epochs", type=int, default=6, help="final fine-tune epochs (mask kept)")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(CKPT_DIR / "resnet18_unstructured_pruned.pth"))
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    device = get_device(args.device)
    data = Cifar10Data()
    model = load_float_model(Path(args.fp32), device)

    base_acc = evaluate(model, data, device)
    banner(f"Unstructured iterative pruning | device={device} | baseline acc {base_acc:.2f}%")
    targets = cubic_schedule(args.sparsity, args.iterations)
    print("cumulative sparsity schedule:", [f"{t:.1%}" for t in targets])

    history, log = [], []
    prev = 0.0
    for it, target in enumerate(targets, start=1):
        # `amount` is relative to the weights that are still alive -> convert cumulative -> incremental
        amount = (target - prev) / (1.0 - prev)
        prune.global_unstructured(prunable_params(model), pruning_method=prune.L1Unstructured, amount=amount)
        s = sparsity_of(model)
        acc_pruned = evaluate(model, data, device)
        print(f"\n--- iteration {it}/{args.iterations}: target {target:.1%} | achieved {s:.2%} | "
              f"accuracy right after pruning {acc_pruned:.2f}%")
        acc_ft = finetune(model, data, device, args.ft_epochs_per_iter, lr=args.lr,
                          tag=f"iter{it}", log=log)
        history.append((it, target, s, acc_pruned, acc_ft))
        prev = target

    print(f"\n--- final fine-tuning ({args.final_ft_epochs} epochs, mask kept) ---")
    final_acc = finetune(model, data, device, args.final_ft_epochs, lr=args.lr, tag="final", log=log)

    banner("Making pruning permanent (prune.remove)")
    sp_before = sparsity_of(model)
    for m, name in prunable_params(model):
        prune.remove(m, name)                       # weight := weight_orig * weight_mask, buffers dropped
    assert not any(hasattr(m, "weight_mask") for m, _ in prunable_params(model)), "mask still attached"
    sp_after = sparsity_of(model)
    perm_acc = evaluate(model, data, device)
    print(f"sparsity before remove: {sp_before:.4%} | after remove: {sp_after:.4%}")
    print(f"accuracy after remove : {perm_acc:.2f}%  (must equal {final_acc:.2f}%)")

    nz, tot = count_nonzero_weights(model)
    print("\nPer-layer sparsity:")
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            z = (m.weight == 0).float().mean().item()
            print(f"  {name:28s} {tuple(m.weight.shape)!s:22s} sparsity {z:6.1%}")

    out = Path(args.out)
    save_checkpoint(out, model, test_acc=perm_acc, sparsity=sp_after, method="global L1 unstructured",
                    iterations=args.iterations)
    print(f"\nnon-zero weights: {nz:,} / {tot:,}  ({sp_after:.2%} sparse)")
    print(f"accuracy: baseline {base_acc:.2f}% -> pruned {perm_acc:.2f}%  ({perm_acc - base_acc:+.2f} pts)")
    print(f"saved -> {out}  ({file_size_mb(out):.2f} MB raw, {gzip_size_mb(out):.2f} MB gzip)")

    with open(LOG_DIR / "unstructured_pruning_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "target_sparsity", "achieved_sparsity", "acc_after_prune", "acc_after_finetune"])
        for row in history:
            w.writerow([row[0], f"{row[1]:.4f}", f"{row[2]:.4f}", f"{row[3]:.2f}", f"{row[4]:.2f}"])
        w.writerow(["final", f"{args.sparsity:.4f}", f"{sp_after:.4f}", "", f"{perm_acc:.2f}"])


if __name__ == "__main__":
    main()
