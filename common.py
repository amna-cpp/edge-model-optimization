"""
common.py - shared utilities for Assignment 03 (Edge Computing & Intelligent Systems)

Contains
  * paths / constants
  * in-memory CIFAR-10 data pipeline (GPU-side augmentation, no DataLoader overhead)
  * a CIFAR-style ResNet-18 that is
        - fusable (Conv+BN+ReLU) for PTQ / QAT,
        - reconstructable with a custom number of internal channels per block
          (needed by the structured-pruning pipeline)
  * train / evaluate helpers
  * model statistics (parameters, non-zeros, MACs, size, latency)
"""
from __future__ import annotations

import gzip
import io
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.quantized import FloatFunctional
from torch.ao.quantization import DeQuantStub, QuantStub, fuse_modules, fuse_modules_qat

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
ONNX_DIR = ROOT / "onnx_models"
LOG_DIR = ROOT / "logs"

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Quantisation backend. On Apple-silicon / ARM CPUs only QNNPACK is available
# (FBGEMM is x86-only).  `select_quant_backend` picks the right one at runtime.
SEED = 42


def ensure_dirs() -> None:
    for d in (DATA_DIR, CKPT_DIR, RESULTS_DIR, ONNX_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Training device: Apple MPS / CUDA if present, otherwise CPU."""
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def select_quant_backend() -> str:
    """Configure torch.backends.quantized.engine (fbgemm on x86, qnnpack on ARM)."""
    engines = torch.backends.quantized.supported_engines
    backend = "fbgemm" if "fbgemm" in engines else "qnnpack"
    torch.backends.quantized.engine = backend
    return backend


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class Cifar10Data:
    """Whole CIFAR-10 held in memory as uint8 tensors; batches are normalised
    (and augmented for training) on the target device."""

    def __init__(self, root: Path = DATA_DIR):
        from torchvision.datasets import CIFAR10  # local import: keeps ONNX scripts light

        train = CIFAR10(str(root), train=True, download=True)
        test = CIFAR10(str(root), train=False, download=True)
        self.x_train = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous()
        self.y_train = torch.tensor(train.targets, dtype=torch.long)
        self.x_test = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous()
        self.y_test = torch.tensor(test.targets, dtype=torch.long)
        self.mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)

    # -- helpers ------------------------------------------------------------ #
    def normalize(self, x_u8: torch.Tensor, device: torch.device) -> torch.Tensor:
        x = x_u8.to(device).float().div_(255.0)
        return (x - self.mean.to(device)) / self.std.to(device)

    @staticmethod
    def _augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
        """Random crop (4px zero padding) + horizontal flip, per sample."""
        n, c, h, w = x.shape
        dev = x.device
        xp = F.pad(x, (pad, pad, pad, pad))
        i = torch.randint(0, 2 * pad + 1, (n,), device=dev)
        j = torch.randint(0, 2 * pad + 1, (n,), device=dev)
        rows = (i[:, None] + torch.arange(h, device=dev)[None])[:, None, :, None]
        cols = (j[:, None] + torch.arange(w, device=dev)[None])[:, None, None, :]
        out = xp[torch.arange(n, device=dev)[:, None, None, None],
                 torch.arange(c, device=dev)[None, :, None, None], rows, cols]
        flip = torch.rand(n, device=dev) < 0.5
        return torch.where(flip[:, None, None, None], out.flip(3), out)

    # -- iterators ---------------------------------------------------------- #
    def train_batches(self, batch_size: int, device: torch.device, augment: bool = True,
                      max_images: int | None = None):
        n = len(self.y_train)
        perm = torch.randperm(n)
        if max_images is not None:
            perm = perm[:max_images]
        for s in range(0, len(perm) - batch_size + 1, batch_size):  # drop last (BN stability)
            idx = perm[s:s + batch_size]
            x = self.normalize(self.x_train[idx], device)
            if augment:
                x = self._augment(x)
            yield x, self.y_train[idx].to(device)

    def test_batches(self, batch_size: int, device: torch.device, limit: int | None = None):
        n = len(self.y_test) if limit is None else min(limit, len(self.y_test))
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            yield self.normalize(self.x_test[s:e], device), self.y_test[s:e].to(device)

    def calibration_batches(self, num_images: int, batch_size: int = 100,
                            device: torch.device = torch.device("cpu"), seed: int = SEED):
        """Fixed random *training* subset (no augmentation) used for PTQ calibration."""
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(self.y_train), generator=g)[:num_images]
        for s in range(0, len(idx), batch_size):
            sl = idx[s:s + batch_size]
            yield self.normalize(self.x_train[sl], device), self.y_train[sl].to(device)


# --------------------------------------------------------------------------- #
# Model: CIFAR ResNet-18 (3x3 stem, no max-pool) with eager-mode quantisation support
# --------------------------------------------------------------------------- #
DEFAULT_PLANES = [64, 64, 128, 128, 256, 256, 512, 512]   # output channels of the 8 blocks


class BasicBlock(nn.Module):
    """conv-bn-relu -> conv-bn  (+ shortcut) -> relu.

    `mid_planes` is the width between the two convolutions.  Structured pruning
    shrinks only this internal width, so the residual-stream width (`planes`) is
    unchanged and the addition with the shortcut always stays dimensionally valid.
    """

    def __init__(self, in_planes: int, mid_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, mid_planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(mid_planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                                          nn.BatchNorm2d(planes))
        # quantisation-friendly residual add + ReLU (a plain `+` cannot be quantised)
        self.skip_add = FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.skip_add.add_relu(out, self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10, mid_channels: list | None = None):
        super().__init__()
        mids = list(mid_channels) if mid_channels is not None else list(DEFAULT_PLANES)
        assert len(mids) == 8, "ResNet-18 has 8 basic blocks"
        self.mid_channels = mids
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        blocks, in_planes, k = [], 64, 0
        for planes, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            stage = []
            for s in (stride, 1):
                stage.append(BasicBlock(in_planes, mids[k], planes, s))
                in_planes = planes
                k += 1
            blocks.append(nn.Sequential(*stage))
        self.layer1, self.layer2, self.layer3, self.layer4 = blocks
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        x = self.fc(x)
        return self.dequant(x)

    def fuse_model(self, qat: bool = False) -> "ResNet18":
        """Fuse Conv+BN(+ReLU) groups. PTQ needs eval() mode, QAT needs train() mode."""
        fuse = fuse_modules_qat if qat else fuse_modules
        fuse(self, [["conv1", "bn1", "relu"]], inplace=True)
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in stage:
                groups = [["conv1", "bn1", "relu1"], ["conv2", "bn2"]]
                if len(block.shortcut) > 0:
                    groups.append(["shortcut.0", "shortcut.1"])
                fuse(block, groups, inplace=True)
        return self


def build_resnet18(mid_channels: list | None = None) -> ResNet18:
    return ResNet18(10, mid_channels)


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, model: nn.Module, **meta) -> None:
    """Save a float model: state_dict (CPU) + the architecture config needed to rebuild it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"state_dict": sd, "mid_channels": getattr(model, "mid_channels", None),
                "meta": meta}, path)


def load_float_model(path: Path, device: str | torch.device = "cpu") -> ResNet18:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_resnet18(ck.get("mid_channels"))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train_one_epoch(model: nn.Module, data: Cifar10Data, optimizer, device: torch.device,
                    scheduler=None, batch_size: int = 128, max_images: int | None = None,
                    label_smoothing: float = 0.0, step_hook=None):
    """One epoch of SGD. Returns (mean loss, train accuracy %)."""
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0
    for x, y in data.train_batches(batch_size, device, augment=True, max_images=max_images):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = F.cross_entropy(out, y, label_smoothing=label_smoothing)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step_hook is not None:
            step_hook()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
    return loss_sum / seen, 100.0 * correct / seen


@torch.no_grad()
def evaluate(model: nn.Module, data: Cifar10Data, device: torch.device,
             batch_size: int = 500, return_preds: bool = False):
    """Accuracy (%) on the full CIFAR-10 test set (optionally with predictions/confidences)."""
    model.eval()
    correct, total = 0, 0
    preds, confs = [], []
    for x, y in data.test_batches(batch_size, device):
        prob = model(x).float().softmax(1)
        conf, pred = prob.max(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        if return_preds:
            preds.append(pred.cpu())
            confs.append(conf.cpu())
    acc = 100.0 * correct / total
    if return_preds:
        return acc, torch.cat(preds).numpy(), torch.cat(confs).numpy()
    return acc


# --------------------------------------------------------------------------- #
# Model statistics
# --------------------------------------------------------------------------- #
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_nonzero_weights(model: nn.Module) -> tuple[int, int]:
    """(non-zero, total) over Conv2d/Linear weights - the tensors that pruning acts on."""
    nz = tot = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nz += int((m.weight != 0).sum().item())
            tot += m.weight.numel()
    return nz, tot


def estimate_macs(model: nn.Module, input_size=(1, 3, 32, 32)) -> int:
    """Multiply-accumulate count of Conv2d / Linear layers for one image."""
    macs = [0]

    def conv_hook(m, inp, out):
        macs[0] += out.numel() // out.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def fc_hook(m, inp, out):
        macs[0] += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(fc_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_size))
    model.train(was_training)
    for h in hooks:
        h.remove()
    return macs[0]


def file_size_mb(path: Path) -> float:
    return os.path.getsize(path) / 1e6


def gzip_size_mb(path: Path) -> float:
    """Size of the file after gzip - shows what zero-valued (sparse) weights buy you."""
    buf = io.BytesIO()
    with open(path, "rb") as f, gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as gz:
        gz.write(f.read())
    return len(buf.getvalue()) / 1e6


@torch.no_grad()
def measure_latency(fn, batch_size: int = 1, warmup: int = 30, runs: int = 200) -> dict:
    """Wall-clock latency of `fn(x)` (x: random CIFAR-sized batch, CPU) in milliseconds."""
    x = torch.randn(batch_size, 3, 32, 32)
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.array(times)
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)),
            "p95_ms": float(np.percentile(t, 95))}


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78, flush=True)
