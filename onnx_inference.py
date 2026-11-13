"""
onnx_inference.py - Deliverable 8 (part 2): CPU inference + benchmark of the exported ONNX models.

Uses only onnxruntime + numpy (no PyTorch needed for inference, apart from loading CIFAR-10 for
the accuracy sweep).

    python onnx_inference.py                       # accuracy + latency of all five ONNX models
    python onnx_inference.py --image cat.png       # classify an image with every ONNX model
    python onnx_inference.py --models structured --image cat.png

Output: results/onnx_benchmark.csv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import CIFAR_MEAN, CIFAR_STD, CLASSES, ONNX_DIR, RESULTS_DIR, banner, ensure_dirs

ONNX_MODELS = {
    "fp32":         ("FP32 baseline",       ONNX_DIR / "resnet18_fp32.onnx"),
    "ptq":          ("PTQ INT8 (QDQ)",      ONNX_DIR / "resnet18_ptq_int8.onnx"),
    "qat":          ("QAT INT8 (QDQ)",      ONNX_DIR / "resnet18_qat_int8.onnx"),
    "unstructured": ("Unstructured pruned", ONNX_DIR / "resnet18_unstructured_pruned.onnx"),
    "structured":   ("Structured pruned",   ONNX_DIR / "resnet18_structured_pruned.onnx"),
}
_MEAN = np.array(CIFAR_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array(CIFAR_STD, dtype=np.float32).reshape(1, 3, 1, 1)


def make_session(path: Path, threads: int = 1) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def preprocess_image(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return (x - _MEAN) / _STD


def run(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    return sess.run(None, {sess.get_inputs()[0].name: x})[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="all")
    ap.add_argument("--image", nargs="*", help="classify these image files instead of benchmarking")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--latency-runs", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    keys = list(ONNX_MODELS) if args.models == "all" else [k.strip() for k in args.models.split(",")]
    sessions = {k: make_session(ONNX_MODELS[k][1], args.threads) for k in keys}

    if args.image:                                     # ---------------- image classification mode
        x = np.concatenate([preprocess_image(p) for p in args.image])
        for k in keys:
            prob = softmax(run(sessions[k], x))
            for p, pr in zip(args.image, prob):
                c = int(pr.argmax())
                print(f"{ONNX_MODELS[k][0]:22s} {Path(p).name:20s} -> {CLASSES[c]:10s} ({pr[c]:.1%})")
        return

    # ---------------------------------------------------- benchmark mode
    from common import Cifar10Data
    ensure_dirs()
    data = Cifar10Data()
    xt = ((data.x_test.numpy().astype(np.float32) / 255.0) - _MEAN) / _STD
    yt = data.y_test.numpy()
    if args.limit:
        xt, yt = xt[:args.limit], yt[:args.limit]
    rng = np.random.default_rng(0)
    one = rng.standard_normal((1, 3, 32, 32)).astype(np.float32)

    rows, fp32_logits = [], None
    for k in keys:
        name, path = ONNX_MODELS[k]
        banner(f"ONNX Runtime (CPU, {args.threads} thread): {name}  [{path.name}]")
        sess = sessions[k]
        t0 = time.perf_counter()
        logits = np.concatenate([run(sess, xt[i:i + 250]) for i in range(0, len(xt), 250)])
        sweep_s = time.perf_counter() - t0
        acc = 100.0 * float((logits.argmax(1) == yt).mean())
        if k == "fp32":
            fp32_logits = logits
        agree = 100.0 * float((logits.argmax(1) == fp32_logits.argmax(1)).mean()) if fp32_logits is not None else 100.0
        for _ in range(30):
            run(sess, one)
        ts = []
        for _ in range(args.latency_runs):
            t = time.perf_counter()
            run(sess, one)
            ts.append((time.perf_counter() - t) * 1000)
        ts = np.array(ts)
        rows.append({"model": name, "key": k, "accuracy_pct": round(acc, 2),
                     "agreement_with_fp32_pct": round(agree, 2),
                     "size_mb": round(path.stat().st_size / 1e6, 2),
                     "latency_bs1_mean_ms": round(float(ts.mean()), 3),
                     "latency_bs1_p50_ms": round(float(np.percentile(ts, 50)), 3),
                     "latency_bs1_p95_ms": round(float(np.percentile(ts, 95)), 3),
                     "throughput_img_s": round(len(xt) / sweep_s, 1), "threads": args.threads,
                     "onnx_file": path.name})
        print(f"accuracy {acc:.2f}% | agreement with FP32 ONNX {agree:.2f}% | "
              f"latency {ts.mean():.2f} ms | throughput {len(xt) / sweep_s:.0f} img/s | {path.stat().st_size / 1e6:.2f} MB")

    import pandas as pd
    df = pd.DataFrame(rows)
    base = df.loc[df.key == "fp32"].iloc[0] if (df.key == "fp32").any() else None
    if base is not None:
        df["speedup_vs_fp32_bs1"] = (base.latency_bs1_mean_ms / df.latency_bs1_mean_ms).round(2)
    df.to_csv(RESULTS_DIR / "onnx_benchmark.csv", index=False)
    banner("ONNX SUMMARY")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.drop(columns=["key", "threads"]).to_string(index=False))
    print(f"\nwritten: {RESULTS_DIR / 'onnx_benchmark.csv'}")


if __name__ == "__main__":
    main()
