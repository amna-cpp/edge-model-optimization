"""
inference.py - Deliverable 6: one inference pipeline for all five models.

    FP32 baseline | PTQ-INT8 | QAT-INT8 | unstructured-pruned | structured-pruned

Examples
    python inference.py                                   # 8 random CIFAR-10 test images, all models
    python inference.py --models fp32,ptq,qat --samples 16
    python inference.py --image cat.png dog.jpg           # any image file (resized to 32x32)
    python inference.py --models structured --image cat.png

Importable API (used by benchmark.py):
    load_model(key) -> torch.nn.Module (CPU, eval)      predict(model, x) -> (labels, probabilities)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import (CIFAR_MEAN, CIFAR_STD, CKPT_DIR, CLASSES, Cifar10Data, load_float_model,
                    select_quant_backend)
from quant_utils import load_int8_torchscript

# key -> (display name, checkpoint path, kind)
MODELS = {
    "fp32":         ("FP32 baseline",        CKPT_DIR / "resnet18_fp32.pth",                   "float"),
    "ptq":          ("PTQ INT8",             CKPT_DIR / "resnet18_ptq_int8_ts.pt",             "int8"),
    "qat":          ("QAT INT8",             CKPT_DIR / "resnet18_qat_int8_ts.pt",             "int8"),
    "unstructured": ("Unstructured pruned",  CKPT_DIR / "resnet18_unstructured_pruned.pth",    "float"),
    "structured":   ("Structured pruned",    CKPT_DIR / "resnet18_structured_pruned.pth",      "float"),
}


def load_model(key: str) -> torch.nn.Module:
    """Load any of the five deployed models on CPU, ready for inference."""
    if key not in MODELS:
        raise KeyError(f"unknown model '{key}', choose from {list(MODELS)}")
    _, path, kind = MODELS[key]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the corresponding training/pruning script first")
    if kind == "int8":
        return load_int8_torchscript(path)         # sets the quantisation backend first
    select_quant_backend()
    return load_float_model(path, "cpu")


def preprocess_image(path: str) -> torch.Tensor:
    """Image file -> normalised 1x3x32x32 tensor."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255.0)
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor):
    """x: N x 3 x 32 x 32 normalised float tensor -> (predicted class ids, probabilities)."""
    prob = model(x).float().softmax(1)
    return prob.argmax(1), prob


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="all", help="comma list of: " + ",".join(MODELS) + " (default all)")
    ap.add_argument("--image", nargs="*", help="image file(s) to classify; default: CIFAR-10 test samples")
    ap.add_argument("--samples", type=int, default=8, help="number of random test images when no --image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    keys = list(MODELS) if args.models == "all" else [k.strip() for k in args.models.split(",")]
    models = {k: load_model(k) for k in keys}

    if args.image:
        xs = torch.cat([preprocess_image(p) for p in args.image])
        names, truth = list(args.image), [None] * len(args.image)
    else:
        data = Cifar10Data()
        idx = torch.randperm(len(data.y_test), generator=torch.Generator().manual_seed(args.seed))[:args.samples]
        xs = data.normalize(data.x_test[idx], torch.device("cpu"))
        names = [f"test[{i}]" for i in idx.tolist()]
        truth = [CLASSES[data.y_test[i]] for i in idx]

    header = f"{'input':14s} {'truth':11s}" + "".join(f"| {MODELS[k][0]:^24s}" for k in keys)
    print(header + "\n" + "-" * len(header))
    results = {k: predict(m, xs) for k, m in models.items()}
    for i, (name, t) in enumerate(zip(names, truth)):
        row = f"{Path(name).name[:14]:14s} {t or '-':11s}"
        for k in keys:
            pred, prob = results[k]
            p = int(pred[i])
            mark = "" if t is None else (" ok" if CLASSES[p] == t else " X ")
            row += f"| {CLASSES[p]:>10s} {prob[i, p].item():6.1%}{mark:>6s}   "
        print(row)


if __name__ == "__main__":
    main()
