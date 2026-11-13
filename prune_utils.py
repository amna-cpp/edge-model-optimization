"""
prune_utils.py - fine-tuning helper shared by the two pruning pipelines.
"""
from __future__ import annotations

import torch

from common import Cifar10Data, evaluate, train_one_epoch


def finetune(model: torch.nn.Module, data: Cifar10Data, device: torch.device, epochs: int,
             lr: float = 0.01, weight_decay: float = 5e-4, batch_size: int = 128,
             tag: str = "fine-tune", log=None) -> float:
    """Fine-tune a (pruned) model with SGD + cosine LR decay; returns final test accuracy (%).

    For masked (unstructured) models the pruning mask is part of the forward pass, so pruned
    weights stay exactly zero throughout fine-tuning.
    """
    steps = (len(data.y_train) // batch_size) * epochs
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay,
                          nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    acc = evaluate(model, data, device)
    for ep in range(1, epochs + 1):
        loss, tr = train_one_epoch(model, data, opt, device, sched, batch_size)
        acc = evaluate(model, data, device)
        print(f"    [{tag}] epoch {ep}/{epochs} | loss {loss:.4f} | train {tr:.2f}% | test {acc:.2f}%",
              flush=True)
        if log is not None:
            log.append({"stage": tag, "epoch": ep, "train_loss": loss, "train_acc": tr, "test_acc": acc})
    return acc
