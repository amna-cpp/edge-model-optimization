"""
make_report_assets.py - builds the figures / console screenshots used in the technical report.

Reads logs/*.csv, logs/*.log and results/*.csv, writes PNGs to report/figures/.
    python make_report_assets.py
"""
from __future__ import annotations

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import LOG_DIR, RESULTS_DIR, ROOT

FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COLORS = {"FP32 baseline": "#4C72B0", "PTQ INT8": "#DD8452", "QAT INT8": "#55A868",
          "Unstructured pruned": "#C44E52", "Structured pruned": "#8172B3"}


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=160)
    plt.close()
    print("wrote", FIG / name)


def training_curves() -> None:
    p = LOG_DIR / "baseline_train_log.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(df.epoch, df.train_loss, marker="o", ms=3)
    ax[0].set(title="Baseline: training loss", xlabel="epoch", ylabel="cross-entropy")
    ax[1].plot(df.epoch, df.train_acc, label="train", marker="o", ms=3)
    ax[1].plot(df.epoch, df.test_acc, label="test", marker="o", ms=3)
    ax[1].set(title="Baseline: accuracy (%)", xlabel="epoch", ylabel="%")
    ax[1].legend()
    for a in ax:
        a.grid(alpha=.3)
    savefig("baseline_training.png")


def model_comparison() -> None:
    p = RESULTS_DIR / "benchmark_results.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    metrics = [("accuracy_pct", "Test accuracy (%)"), ("size_mb", "Checkpoint size (MB)"),
               ("latency_bs1_mean_ms", "Latency, batch 1 (ms, 1 thread)"),
               ("throughput_bs128_img_s", "Throughput, batch 128 (img/s)")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (col, title) in zip(axes, metrics):
        bars = ax.bar(range(len(df)), df[col], color=[COLORS[m] for m in df.model])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([m.replace(" ", "\n") for m in df.model], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.bar_label(bars, fmt="%.2f" if col != "throughput_bs128_img_s" else "%.0f", fontsize=8)
        if col == "accuracy_pct":
            ax.set_ylim(max(0, df[col].min() - 3), min(100, df[col].max() + 1.5))
        ax.grid(axis="y", alpha=.3)
    savefig("model_comparison.png")


def onnx_comparison() -> None:
    p = RESULTS_DIR / "onnx_benchmark.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (col, title) in zip(axes, [("accuracy_pct", "ONNX accuracy (%)"),
                                       ("latency_bs1_mean_ms", "ONNX latency, batch 1 (ms)"),
                                       ("size_mb", "ONNX file size (MB)")]):
        bars = ax.bar(range(len(df)), df[col], color=[COLORS[m.split(" (")[0]] if m.split(" (")[0] in COLORS else "#888" for m in df.model])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([m.replace(" ", "\n") for m in df.model], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        if col == "accuracy_pct":
            ax.set_ylim(max(0, df[col].min() - 3), min(100, df[col].max() + 1.5))
        ax.grid(axis="y", alpha=.3)
    savefig("onnx_comparison.png")


def pruning_plots() -> None:
    p = LOG_DIR / "unstructured_pruning_log.csv"
    if p.exists():
        df = pd.read_csv(p)
        df = df[df.iteration != "final"].astype({"achieved_sparsity": float, "acc_after_prune": float,
                                                 "acc_after_finetune": float})
        plt.figure(figsize=(6, 3.8))
        plt.plot(df.achieved_sparsity * 100, df.acc_after_prune, "o--", label="right after pruning")
        plt.plot(df.achieved_sparsity * 100, df.acc_after_finetune, "o-", label="after fine-tuning")
        plt.xlabel("global sparsity (%)")
        plt.ylabel("test accuracy (%)")
        plt.title("Iterative unstructured pruning")
        plt.grid(alpha=.3)
        plt.legend()
        savefig("unstructured_iterations.png")
    p = LOG_DIR / "structured_pruning_log.csv"
    if p.exists():
        df = pd.read_csv(p)
        plt.figure(figsize=(6, 3.8))
        plt.plot(df.epoch, df.test_acc, "o-")
        plt.xlabel("fine-tune epoch")
        plt.ylabel("test accuracy (%)")
        plt.title("Structured pruning: fine-tuning recovery")
        plt.grid(alpha=.3)
        savefig("structured_finetune.png")
    # channel widths before / after
    try:
        from common import DEFAULT_PLANES, CKPT_DIR
        import torch
        ck = torch.load(CKPT_DIR / "resnet18_structured_pruned.pth", map_location="cpu", weights_only=False)
        mids = ck["mid_channels"]
        x = range(8)
        plt.figure(figsize=(7, 3.8))
        plt.bar([i - .2 for i in x], DEFAULT_PLANES, .4, label="original", color="#4C72B0")
        plt.bar([i + .2 for i in x], mids, .4, label="pruned", color="#8172B3")
        plt.xticks(list(x), [f"L{s}.{b}" for s in range(1, 5) for b in range(2)])
        plt.ylabel("channels between conv1 and conv2")
        plt.title("Structured pruning: block widths")
        plt.legend()
        savefig("structured_channels.png")
    except Exception as exc:   # pragma: no cover
        print("channel plot skipped:", exc)


def console_shots() -> None:
    """Render saved console logs as terminal-style PNG 'screenshots'."""
    for log, title, tail in [("01_train_baseline", "Baseline training", 26), ("02_ptq", "PTQ", 40),
                             ("03_qat", "QAT", 40), ("04_unstructured", "Unstructured pruning", 48),
                             ("05_structured", "Structured pruning", 48), ("06_inference_demo", "Inference demo", 14),
                             ("07_benchmark", "Benchmark summary", 16), ("08_export_onnx", "ONNX export", 30),
                             ("09_onnx_benchmark", "ONNX benchmark", 16)]:
        p = LOG_DIR / f"{log}.log"
        if not p.exists():
            continue
        lines = p.read_text().rstrip().splitlines()[-tail:]
        lines = [textwrap.shorten(l, 150, placeholder=" ...") if len(l) > 150 else l for l in lines]
        h = max(1.6, 0.122 * len(lines) + 0.75)
        fig = plt.figure(figsize=(13, h), facecolor="#1e1e1e")
        fig.text(0.01, 1 - 0.25 / h, f"$ python {log.split('_', 1)[1]}.py   [{title}]", color="#9cdcfe",
                 family="monospace", fontsize=8.5, va="top")
        fig.text(0.01, 1 - 0.55 / h, "\n".join(lines), color="#d4d4d4", family="monospace", fontsize=7.5, va="top")
        plt.savefig(FIG / f"console_{log}.png", dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        print("wrote", FIG / f"console_{log}.png")


if __name__ == "__main__":
    training_curves()
    model_comparison()
    onnx_comparison()
    pruning_plots()
    console_shots()
