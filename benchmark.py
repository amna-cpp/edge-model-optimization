"""
benchmark.py - Deliverable 7: automatically evaluate all five models on the CIFAR-10 test set.

    python benchmark.py                 # accuracy on all 10,000 test images + CPU speed
    python benchmark.py --threads 4

Measures, per model
    accuracy, parameters, non-zero weights/sparsity, MACs, checkpoint size (raw and gzip),
    batch-1 latency (mean / p50 / p95), batch-128 throughput, and per-sample predictions.

Outputs (results/)
    benchmark_results.csv                 summary - all five models
    predictions.csv                       per-image predictions + confidence - all five models
    q1_quantization_results.csv           FP32 vs PTQ vs QAT summary
    q1_quantization_predictions.csv       per-image predictions of those three
    q2_pruning_results.csv                FP32 vs unstructured vs structured summary

All timing is done on CPU (the INT8 kernels only exist for CPU) with a fixed thread count.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from common import (CLASSES, RESULTS_DIR, Cifar10Data, banner, count_nonzero_weights,
                    count_parameters, ensure_dirs, estimate_macs, evaluate, file_size_mb,
                    gzip_size_mb, load_float_model, measure_latency, set_seed)
from inference import MODELS, load_model


def throughput(model, data: Cifar10Data, batch_size: int = 128, batches: int = 6) -> float:
    """Images per second over real test batches (pre-normalised so only the model is timed)."""
    cpu = torch.device("cpu")
    xs = [x for x, _ in data.test_batches(batch_size, cpu, limit=batch_size * batches)]
    with torch.no_grad():
        model(xs[0])                                   # warm-up
        t0 = time.perf_counter()
        for x in xs:
            model(x)
        dt = time.perf_counter() - t0
    return batch_size * len(xs) / dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=1, help="CPU threads for the timing runs (default 1)")
    ap.add_argument("--eval-threads", type=int, default=4, help="CPU threads for the accuracy sweep")
    ap.add_argument("--latency-runs", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N test images (debug)")
    args = ap.parse_args()

    ensure_dirs()
    set_seed()
    cpu = torch.device("cpu")
    data = Cifar10Data()
    if args.limit:
        data.x_test, data.y_test = data.x_test[:args.limit], data.y_test[:args.limit]

    rows, preds = [], {"index": np.arange(len(data.y_test)), "label": data.y_test.numpy(),
                       "label_name": [CLASSES[i] for i in data.y_test]}
    fp32_float = load_float_model(MODELS["fp32"][1], cpu)

    for key, (name, path, kind) in MODELS.items():
        banner(f"Benchmarking: {name}  [{path.name}]")
        model = load_model(key)
        # architecture reference (for params / MACs); INT8 models share the FP32 architecture
        arch = fp32_float if kind == "int8" else model
        nz, tot = count_nonzero_weights(arch)

        torch.set_num_threads(args.eval_threads)
        acc, pred, conf = evaluate(model, data, cpu, return_preds=True)
        preds[f"{key}_pred"] = pred
        preds[f"{key}_conf"] = np.round(conf, 4)

        torch.set_num_threads(args.threads)
        lat = measure_latency(model, 1, runs=args.latency_runs)
        thr = throughput(model, data)

        # size of the *state-dict* checkpoint (INT8: the .pth next to the TorchScript file)
        size_path = path.with_name(path.name.replace("_ts.pt", ".pth")) if kind == "int8" else path
        row = {
            "model": name, "key": key, "accuracy_pct": round(acc, 2),
            "params": count_parameters(arch), "nonzero_weights": nz, "total_weights": tot,
            "sparsity_pct": round(100 * (1 - nz / tot), 2),
            "macs_M": round(estimate_macs(arch) / 1e6, 1),
            "size_mb": round(file_size_mb(size_path), 2), "size_gzip_mb": round(gzip_size_mb(size_path), 2),
            "latency_bs1_mean_ms": round(lat["mean_ms"], 3), "latency_bs1_p50_ms": round(lat["p50_ms"], 3),
            "latency_bs1_p95_ms": round(lat["p95_ms"], 3), "throughput_bs128_img_s": round(thr, 1),
            "timing_threads": args.threads, "checkpoint": path.name,
        }
        rows.append(row)
        print(f"accuracy {acc:.2f}% | latency(bs=1) {lat['mean_ms']:.2f} ms (p95 {lat['p95_ms']:.2f}) | "
              f"throughput {thr:.0f} img/s | size {row['size_mb']} MB | sparsity {row['sparsity_pct']}%")

    df = pd.DataFrame(rows)
    base = df.loc[df.key == "fp32"].iloc[0]
    df["acc_drop_vs_fp32_pts"] = (base.accuracy_pct - df.accuracy_pct).round(2)
    df["speedup_vs_fp32_bs1"] = (base.latency_bs1_mean_ms / df.latency_bs1_mean_ms).round(2)
    df["speedup_vs_fp32_throughput"] = (df.throughput_bs128_img_s / base.throughput_bs128_img_s).round(2)
    df["size_reduction_x"] = (base.size_mb / df.size_mb).round(2)
    df["gzip_size_reduction_x"] = (base.size_gzip_mb / df.size_gzip_mb).round(2)
    pdf = pd.DataFrame(preds)

    df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
    pdf.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    q1, q2 = ["fp32", "ptq", "qat"], ["fp32", "unstructured", "structured"]
    df[df.key.isin(q1)].to_csv(RESULTS_DIR / "q1_quantization_results.csv", index=False)
    df[df.key.isin(q2)].to_csv(RESULTS_DIR / "q2_pruning_results.csv", index=False)
    base_cols = ["index", "label", "label_name"]
    pdf[base_cols + [f"{k}_{s}" for k in q1 for s in ("pred", "conf")]].to_csv(
        RESULTS_DIR / "q1_quantization_predictions.csv", index=False)

    banner("SUMMARY")
    show = ["model", "accuracy_pct", "acc_drop_vs_fp32_pts", "size_mb", "size_gzip_mb", "params",
            "sparsity_pct", "macs_M", "latency_bs1_mean_ms", "throughput_bs128_img_s",
            "speedup_vs_fp32_bs1"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df[show].to_string(index=False))
    print(f"\nCSV files written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
