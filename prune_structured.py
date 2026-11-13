"""
prune_structured.py - Deliverable 5: structured CHANNEL pruning with architecture reconstruction.

Strategy (dimension-safe for residual networks)
  Every BasicBlock is   x -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> (+ shortcut(x)) -> relu
  The residual-stream width (block output) is tied to the shortcut and to the next block's input,
  so pruning it would force changes across a whole stage. The *internal* width between conv1 and
  conv2 is private to the block, so we prune there:

      * score  : L1 norm of every conv1 filter (output channel)
      * remove : the lowest-scoring filters of conv1, the matching BatchNorm channels of bn1,
                 and the matching INPUT channels of conv2   -> all tensor dims stay consistent
      * rebuild: a *new, physically smaller* ResNet-18 (BasicBlock(in, mid', out)) is created and
                 the surviving weights are copied in. No masks / zero channels remain.

  The rebuilt network is verified to be numerically identical to the original with the pruned
  channels masked to zero, then fine-tuned to recover accuracy.

    python prune_structured.py --ratios 0.25,0.25,0.4,0.4,0.6,0.6,0.7,0.7 --ft-epochs 12
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch

from common import (CKPT_DIR, LOG_DIR, Cifar10Data, banner, build_resnet18, count_parameters,
                    ensure_dirs, estimate_macs, evaluate, file_size_mb, get_device,
                    load_float_model, save_checkpoint, set_seed)
from prune_utils import finetune

MIN_CHANNELS = 8


def blocks_of(model):
    """[(name, block)] for the 8 BasicBlocks in order."""
    return [(f"layer{s}.{b}", getattr(model, f"layer{s}")[b]) for s in range(1, 5) for b in range(2)]


def select_channels(block, ratio: float) -> torch.Tensor:
    """Indices (sorted) of the conv1 filters to KEEP, ranked by L1 norm."""
    scores = block.conv1.weight.detach().abs().sum(dim=(1, 2, 3))
    n_keep = max(MIN_CHANNELS, int(round(scores.numel() * (1.0 - ratio))))
    return scores.topk(n_keep).indices.sort().values


def reconstruct(model, keep: dict):
    """Build a smaller ResNet-18 and copy the surviving weights into it."""
    old_sd = model.state_dict()
    new_mid = [len(keep[name]) for name, _ in blocks_of(model)]
    new_model = build_resnet18(new_mid)
    pat = re.compile(r"^(layer\d\.\d)\.(conv1|bn1|conv2)\.(\w+)$")
    new_sd = {}
    for k, v in old_sd.items():
        m = pat.match(k)
        if m and m.group(1) in keep:
            idx = keep[m.group(1)].to(v.device)
            if m.group(2) in ("conv1", "bn1") and v.ndim > 0:      # out-channels of conv1 / bn1 params
                v = v[idx]
            elif m.group(2) == "conv2":                            # in-channels of conv2
                v = v[:, idx]
        new_sd[k] = v.clone()
    new_model.load_state_dict(new_sd)
    return new_model.to(next(model.parameters()).device)


@torch.no_grad()
def verify_equivalence(orig, pruned, keep: dict, data: Cifar10Data, device) -> float:
    """Max |logit difference| between (original with pruned channels zeroed) and the rebuilt model."""
    hooks = []
    for name, block in blocks_of(orig):
        mask = torch.zeros(block.conv1.out_channels, device=device)
        mask[keep[name].to(device)] = 1.0
        hooks.append(block.relu1.register_forward_hook(
            lambda m, i, o, mask=mask: o * mask.view(1, -1, 1, 1)))
    x, _ = next(data.test_batches(256, device))
    orig.eval(), pruned.eval()
    diff = (orig(x) - pruned(x)).abs().max().item()
    for h in hooks:
        h.remove()
    return diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fp32", default=str(CKPT_DIR / "resnet18_fp32.pth"))
    ap.add_argument("--ratios", default="0.25,0.25,0.4,0.4,0.6,0.6,0.7,0.7",
                    help="fraction of internal channels removed in each of the 8 blocks")
    ap.add_argument("--ft-epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(CKPT_DIR / "resnet18_structured_pruned.pth"))
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    device = get_device(args.device)
    data = Cifar10Data()
    ratios = [float(r) for r in args.ratios.split(",")]
    assert len(ratios) == 8, "need one ratio per basic block (8)"

    model = load_float_model(Path(args.fp32), device)
    base_acc = evaluate(model, data, device)
    base_params, base_macs = count_parameters(model), estimate_macs(model.cpu())
    model.to(device)
    banner(f"Structured channel pruning | device={device} | baseline acc {base_acc:.2f}%")

    # 1. rank + select
    keep = {name: select_channels(block, r) for (name, block), r in zip(blocks_of(model), ratios)}
    print(f"{'block':10s} {'mid channels':>14s} {'kept':>6s} {'removed':>8s}")
    for (name, block), r in zip(blocks_of(model), ratios):
        c = block.conv1.out_channels
        print(f"{name:10s} {c:14d} {len(keep[name]):6d} {c - len(keep[name]):8d}   (ratio {r:.0%})")

    # 2. rebuild a physically smaller network
    pruned = reconstruct(model, keep)
    diff = verify_equivalence(model, pruned, keep, data, device)
    print(f"\nreconstruction check: max |logit diff| vs masked original = {diff:.2e}  "
          f"({'OK' if diff < 1e-3 else 'FAILED'})")
    assert diff < 1e-3, "reconstructed network does not match the masked original"
    print("new internal widths:", pruned.mid_channels)
    acc_no_ft = evaluate(pruned, data, device)
    print(f"accuracy right after pruning (no fine-tuning): {acc_no_ft:.2f}%")

    # 3. fine-tune
    banner(f"Fine-tuning pruned network ({args.ft_epochs} epochs)")
    log = []
    final_acc = finetune(pruned, data, device, args.ft_epochs, lr=args.lr, tag="structured", log=log)

    # 4. report + save
    p_params, p_macs = count_parameters(pruned), estimate_macs(pruned.cpu())
    pruned.to(device)
    out = Path(args.out)
    save_checkpoint(out, pruned, test_acc=final_acc, ratios=ratios, method="L1 channel pruning")
    zero_ch = sum(int((blk.conv1.weight.abs().sum(dim=(1, 2, 3)) == 0).sum()) for _, blk in blocks_of(pruned))
    print(f"\nzero-valued conv1 filters left in rebuilt model: {zero_ch} (no masked channels)")
    print(f"parameters: {base_params:,} -> {p_params:,}  ({1 - p_params / base_params:.1%} fewer)")
    print(f"MACs      : {base_macs / 1e6:.1f}M -> {p_macs / 1e6:.1f}M  ({1 - p_macs / base_macs:.1%} fewer)")
    print(f"accuracy  : {base_acc:.2f}% -> {acc_no_ft:.2f}% (pruned) -> {final_acc:.2f}% (fine-tuned)")
    print(f"saved -> {out}  ({file_size_mb(out):.2f} MB)")

    with open(LOG_DIR / "structured_pruning_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "epoch", "train_loss", "train_acc", "test_acc"])
        w.writeheader()
        w.writerows(log)


if __name__ == "__main__":
    main()
