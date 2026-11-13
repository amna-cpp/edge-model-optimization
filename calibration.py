"""
calibration.py - calibration pipeline for post-training static quantisation.

A prepared model (observers inserted) is fed a small subset of *training* images.
The observers record activation ranges, from which scale / zero-point are derived.
Test data is never used for calibration.
"""
from __future__ import annotations

import torch

from common import Cifar10Data


@torch.no_grad()
def calibrate(prepared_model: torch.nn.Module, data: Cifar10Data, num_images: int = 2000,
              batch_size: int = 100, seed: int = 42) -> int:
    """Run `num_images` training images (no augmentation) through the prepared model.

    Returns the number of images actually used.
    """
    prepared_model.eval()
    used = 0
    for i, (x, _) in enumerate(data.calibration_batches(num_images, batch_size, seed=seed)):
        prepared_model(x)
        used += x.size(0)
        if (i + 1) % 5 == 0:
            print(f"  calibration: {used}/{num_images} images", flush=True)
    return used
