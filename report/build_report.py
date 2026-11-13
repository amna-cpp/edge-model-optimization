"""
build_report.py - assembles report/Technical_Report.docx from the result CSVs, logs and figures
(every number in the tables is read from results/*.csv, so the report cannot drift from the data).

    python report/build_report.py            # then: soffice --headless --convert-to pdf report/Technical_Report.docx
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
RES = ROOT / "results"
LOG = ROOT / "logs"

bench = pd.read_csv(RES / "benchmark_results.csv")
onnx = pd.read_csv(RES / "onnx_benchmark.csv")
B = {r.key: r for r in bench.itertuples()}
O = {r.key: r for r in onnx.itertuples()}

doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2)
    s.top_margin = s.bottom_margin = Cm(2.0)
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10.5)


def H(text, lvl=1):
    doc.add_heading(text, level=lvl)


def P(text, bold=False, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    return p


def BL(items):
    for t in items:
        doc.add_paragraph(t, style="List Bullet")


def table(df: pd.DataFrame, widths=None, font=8.5):
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, c in enumerate(df.columns):
        cell = t.rows[0].cells[i]
        cell.text = str(c)
        for r in cell.paragraphs[0].runs:
            r.bold, r.font.size = True, Pt(font)
    for row in df.itertuples(index=False):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
            for r in cells[i].paragraphs[0].runs:
                r.font.size = Pt(font)
    doc.add_paragraph()


def fig(name, caption, width=16.0):
    p = FIG / name
    if not p.exists():
        return
    doc.add_picture(str(p), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(caption)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.runs[0].italic = True
    c.runs[0].font.size = Pt(9)


# ----------------------------------------------------------------------------- title
t = doc.add_heading("Model Optimisation for Edge Deployment", 0)
P("Quantisation (PTQ, QAT) and Pruning (unstructured, structured) of a CIFAR-10 ResNet-18", bold=True)
P("Assignment 03 — Edge Computing & Intelligent Systems")
P("Platform: Apple M4 (macOS), PyTorch 2.8, Python 3.9, ONNX-Runtime 1.19. All CPU timings use 1 thread unless stated.",
  italic=True)

# ----------------------------------------------------------------------------- 1
H("1. Summary of results")
fp = B["fp32"]
P("All five models were evaluated on the full 10,000-image CIFAR-10 test set with one automated script "
  "(benchmark.py). Sizes are the state-dict checkpoints on disk; latency is batch-1 on one CPU thread; "
  "throughput is batch-128.")
t1 = bench[["model", "accuracy_pct", "acc_drop_vs_fp32_pts", "size_mb", "size_gzip_mb", "params", "sparsity_pct",
            "macs_M", "latency_bs1_mean_ms", "throughput_bs128_img_s", "speedup_vs_fp32_bs1"]].copy()
t1.columns = ["Model", "Acc %", "Drop (pts)", "Size MB", "Gzip MB", "Params", "Sparsity %", "MACs (M)",
              "Lat. bs1 (ms)", "Img/s bs128", "Speed-up"]
table(t1, font=7.5)
BL([
    f"PTQ-INT8: {B['ptq'].size_reduction_x:.2f}x smaller, {B['ptq'].speedup_vs_fp32_bs1:.2f}x faster at batch 1, "
    f"accuracy {B['ptq'].accuracy_pct:.2f}% (FP32 {fp.accuracy_pct:.2f}%).",
    f"QAT-INT8: {B['qat'].size_reduction_x:.2f}x smaller, {B['qat'].speedup_vs_fp32_bs1:.2f}x faster, "
    f"accuracy {B['qat'].accuracy_pct:.2f}%.",
    f"Unstructured pruning (90% sparse): accuracy {B['unstructured'].accuracy_pct:.2f}%, gzip size "
    f"{B['unstructured'].gzip_size_reduction_x:.1f}x smaller, but NO speed-up on a dense CPU runtime "
    f"({B['unstructured'].speedup_vs_fp32_bs1:.2f}x).",
    f"Structured pruning: {100 * (1 - B['structured'].params / fp.params):.0f}% fewer parameters, "
    f"{100 * (1 - B['structured'].macs_M / fp.macs_M):.0f}% fewer MACs, {B['structured'].speedup_vs_fp32_bs1:.2f}x faster, "
    f"accuracy {B['structured'].accuracy_pct:.2f}%.",
])
fig("model_comparison.png", "Figure 1 - accuracy, size, latency and throughput of the five models.", 17)
P("Statistical caveat: with 10,000 test images the binomial standard error of an accuracy near 93% is about "
  "0.25 percentage points, and every result is a single run with a fixed seed. Differences of a few tenths of a "
  "point between the five models (including PTQ appearing 0.10 pts 'better' than FP32) are therefore within noise "
  "and should be read as 'no measurable accuracy loss', not as one method being better.", italic=True)

# ----------------------------------------------------------------------------- 2
H("2. Implementation")
H("2.1 Common infrastructure (common.py)", 2)
BL([
    "Data: CIFAR-10 held in memory as uint8; per-batch normalisation and random-crop (4 px padding) + horizontal-flip "
    "augmentation done on the accelerator. The test set is only used for evaluation, never for training or calibration.",
    "Model: CIFAR-style ResNet-18 (3x3 stem, no max-pool, 11.17 M parameters, 555 M MACs). Every residual add+ReLU "
    "uses FloatFunctional.add_relu so the network can be quantised in eager mode; the block constructor takes the "
    "inner width as an argument so a physically smaller network can be rebuilt after channel pruning.",
    "Reproducibility: fixed seed 42, run_all.sh runs the whole pipeline, every step logs to logs/.",
])
H("2.2 Baseline FP32 (train_baseline.py)", 2)
P("20 epochs, SGD (Nesterov momentum 0.9, weight-decay 5e-4), one-cycle LR schedule with peak 0.1, batch 128, "
  "on the Apple GPU (MPS). The final-epoch model is saved (no best-epoch selection on the test set). "
  f"Result: {fp.accuracy_pct:.2f}% test accuracy (training took 53 minutes).")
fig("baseline_training.png", "Figure 2 - baseline training curves.", 15)

H("2.3 Post-training static quantisation (ptq.py, calibration.py)", 2)
BL([
    "Fusion: Conv+BN+ReLU, Conv+BN (including the 1x1 shortcut convolutions) are fused with fuse_modules (eval mode).",
    "Backend: torch.backends.quantized.engine is chosen at run time - FBGEMM on x86, QNNPACK on this ARM machine "
    "(FBGEMM is not available on Apple silicon).",
    "QConfig: per-channel symmetric INT8 weights, UINT8 activations with a HistogramObserver.",
    "Calibration: 2,000 randomly chosen TRAINING images (no augmentation) are passed through the observed model.",
    "Convert + save: converted to INT8, evaluated, saved both as state_dict and as self-contained TorchScript; the "
    "script reloads the TorchScript file and checks that it reproduces the in-memory accuracy exactly.",
])
P(f"Result: {B['ptq'].accuracy_pct:.2f}% test accuracy, {B['ptq'].size_mb:.2f} MB ({B['ptq'].size_reduction_x:.2f}x smaller).")

H("2.4 Quantisation-aware training (qat.py)", 2)
BL([
    "Initialised from the pretrained FP32 checkpoint; Conv+BN+ReLU fused with the QAT fusion (train mode); "
    "FakeQuantize modules inserted with prepare_qat (QNNPACK QAT qconfig).",
    "Fine-tuning: 3 epochs x 20,000 random training images, SGD lr 0.005 with cosine decay. BatchNorm statistics are "
    "frozen from epoch 2 and the observers from epoch 3. QAT runs on the CPU because the fused fake-quant operator "
    "is not implemented for the Apple GPU (MPS); this limited the budget to ~15 minutes per epoch.",
    "Convert to INT8, evaluate, save state_dict + TorchScript (+ the fake-quant model for reference).",
])
P(f"Result: {B['qat'].accuracy_pct:.2f}% test accuracy, {B['qat'].size_mb:.2f} MB. In this experiment QAT did not beat "
  "PTQ: the FP32 model is quantisation-friendly (per-channel weights + histogram calibration already lose "
  "nothing measurable), so there is little for QAT to recover, and QAT used per-tensor weights and a very short "
  "schedule. We report this as measured rather than tune toward the textbook expectation.")

H("2.5 Unstructured pruning (prune_unstructured.py)", 2)
BL([
    "Global L1-magnitude pruning over all Conv2d and Linear weights (torch.nn.utils.prune), with masks "
    "(weight_orig + weight_mask) active during fine-tuning.",
    "Iterative: 5 pruning iterations on a cubic sparsity schedule (43.9 -> 70.6 -> 84.2 -> 89.3 -> 90.0 %), one "
    "fine-tuning epoch after each, then 6 final fine-tuning epochs.",
    "Permanent: prune.remove bakes the mask into the weight tensors; the script checks that sparsity (90.00%) and accuracy "
    "are unchanged by this step.",
])
P("Observation: global pruning removes far more from the wide, redundant last stage (layer4 convolutions 93-99.9% sparse) "
  "than from the early layers (40-45%), and the first conv and fully-connected layer are barely pruned.")
fig("unstructured_iterations.png", "Figure 3 - accuracy right after each pruning step and after fine-tuning.", 11)

H("2.6 Structured channel pruning (prune_structured.py)", 2)
BL([
    "Scope: the inner width of each of the 8 residual blocks (conv1 output = conv2 input). The residual-stream width is "
    "tied to the shortcut connections and the next block's input, so leaving it untouched keeps every addition "
    "dimensionally valid without touching other layers.",
    "Ranking: L1 norm of each conv1 filter. Pruned fraction per block: 25/25/40/40/60/60/70/70 % (more in the later, wider stages).",
    "Reconstruction: a new, physically smaller ResNet-18 is instantiated with the reduced widths "
    "[48, 48, 77, 77, 102, 102, 154, 154] and the surviving weights (conv1 filters, bn1 channels, conv2 input channels) "
    "are copied in. No masks and no zero-valued channels remain.",
    "Verification: the rebuilt network's logits equal those of the original network with the pruned channels masked "
    "to zero (max difference 0 on a test batch). The script asserts this before fine-tuning.",
    "Fine-tuning: 12 epochs, SGD lr 0.02 with cosine decay. Accuracy fell to 30.2% straight after pruning and recovered "
    f"to {B['structured'].accuracy_pct:.2f}%.",
])
fig("structured_channels.png", "Figure 4 - internal block widths before / after structured pruning.", 11)
fig("structured_finetune.png", "Figure 5 - recovery of the structured-pruned model during fine-tuning.", 11)

H("2.7 Inference pipeline (inference.py)", 2)
P("A single entry point loads any of the five models (FP32 / PTQ-INT8 / QAT-INT8 / unstructured / structured) "
  "through load_model(key): float models are rebuilt from their checkpoint (the structured checkpoint stores its "
  "channel widths), INT8 models are loaded from TorchScript with the right quantisation backend selected first. "
  "It classifies CIFAR-10 test samples or arbitrary image files (resized to 32x32) and prints a side-by-side table.")
fig("console_06_inference_demo.png", "Figure 6 - inference.py on 8 random test images, all five models.", 17)

H("2.8 Benchmark (benchmark.py)", 2)
P("Evaluates all five models on the full test set and writes: results/benchmark_results.csv (summary), "
  "results/predictions.csv (per-image label, prediction and confidence for each model), and the per-question subsets "
  "q1_quantization_results.csv, q1_quantization_predictions.csv, q2_pruning_results.csv. Latency is the mean of 200 "
  "batch-1 runs after 30 warm-up runs (p50/p95 also recorded); throughput is measured over 768 test images at batch 128.")
fig("console_07_benchmark.png", "Figure 7 - benchmark.py console output.", 17)

# ----------------------------------------------------------------------------- 3
H("3. ONNX deployment")
P("export_onnx.py exports all five models; onnx_inference.py runs them with ONNX-Runtime's CPU provider "
  "(no PyTorch needed for inference) and writes results/onnx_benchmark.csv.")
BL([
    "FP32, unstructured and structured models: direct torch.onnx.export (opset 13, dynamic batch).",
    "INT8 models: PyTorch eager-mode quantised operators have no ONNX representation. The converted INT8 model is "
    "therefore re-expressed as a QDQ graph: the exact INT8 weights, per-channel scales, activation scales and "
    "zero-points are read from the converted PyTorch model and written as QuantizeLinear/DequantizeLinear nodes "
    "around Conv/Gemm/Add/ReLU, then weight Quantize nodes are constant-folded to real INT8 initialisers. "
    "The ONNX INT8 models therefore carry the PTQ-calibrated / QAT-learned parameters (not a re-calibration).",
    "Fidelity: the QDQ twin agrees with the PyTorch INT8 model on 99.6% (PTQ) / 99.8% (QAT) of 500 test images; "
    "small differences come from the float emulation of integer arithmetic.",
])
t2 = onnx[["model", "accuracy_pct", "agreement_with_fp32_pct", "size_mb", "latency_bs1_mean_ms", "throughput_img_s",
           "speedup_vs_fp32_bs1"]].copy()
t2.columns = ["ONNX model", "Acc %", "Agreement w/ FP32 %", "Size MB", "Lat. bs1 (ms)", "Img/s", "Speed-up"]
table(t2, font=8)
fig("onnx_comparison.png", "Figure 8 - ONNX-Runtime CPU results.", 15)
P(f"Honest finding: ONNX accuracies match PyTorch ({O['fp32'].accuracy_pct:.2f}% / {O['ptq'].accuracy_pct:.2f}% / "
  f"{O['qat'].accuracy_pct:.2f}% / {O['unstructured'].accuracy_pct:.2f}% / {O['structured'].accuracy_pct:.2f}%). "
  f"The INT8 ONNX files are {O['fp32'].size_mb / O['ptq'].size_mb:.1f}x smaller but are NOT faster under ONNX-Runtime on "
  f"this Apple CPU ({O['ptq'].speedup_vs_fp32_bs1:.2f}x), unlike PyTorch's QNNPACK kernels (1.2x). ONNX-Runtime's "
  "FP32 convolutions are well optimised for this CPU and its QDQ execution adds quantise/dequantise overhead; "
  "INT8 speed-ups depend strongly on the target CPU and runtime. The structured-pruned model gives the largest real "
  f"speed-up ({O['structured'].speedup_vs_fp32_bs1:.2f}x) because it simply has fewer operations.")

# ----------------------------------------------------------------------------- 4
H("4. Discussion")
BL([
    "Quantisation (PTQ/QAT) gives a reliable ~4x reduction in model size with no measurable accuracy change, and a "
    "modest 1.2x speed-up on this CPU in PyTorch. Calibration with 2,000 images was enough; PTQ took about a minute.",
    "Unstructured pruning reaches 90% sparsity with no accuracy loss, but the zeros live inside dense tensors: the file is "
    "the same size and the dense kernels do the same work, so latency does not improve. The benefit shows up only with "
    "compression (gzip: 41.5 -> 7.1 MB) or a sparse-aware runtime/hardware.",
    "Structured pruning delivers genuine compute savings with standard dense kernels (-65% parameters, -47% MACs, "
    "1.5x faster in PyTorch, 1.8x in ONNX-Runtime), at the cost of a larger accuracy hit directly after pruning "
    "(30%) that needs a proper fine-tuning budget; after 12 epochs it is within 0.1 pts of the baseline.",
    "The techniques are complementary (e.g. structured pruning followed by INT8 quantisation) but combining them "
    "was outside the assignment's scope and was not evaluated.",
])
H("5. Limitations")
BL([
    "Single seed, single run per method; the test set doubled as the monitoring set during training (no separate "
    "validation split), though no model selection was done on it (final-epoch models are saved).",
    "Timing is from one Apple M4 laptop, 1 thread, with other applications running; use it for relative comparison only. "
    "INT8 checkpoints are QNNPACK-specific and must be regenerated (ptq.py / qat.py) for an x86 FBGEMM deployment.",
    "QAT used a short schedule (3 x 20k images) because it must run on the CPU; a longer schedule or per-channel QAT "
    "weights may change the PTQ-vs-QAT ordering.",
    "Structured pruning prunes only the inner width of residual blocks; pruning the residual stream would give larger "
    "savings but needs coupled pruning across a whole stage.",
    "The 'screenshots' in the appendix are the saved console logs of each run rendered to images, not operating-system "
    "screen captures.",
])

# ----------------------------------------------------------------------------- 6
H("6. Deliverables map")
dm = pd.DataFrame([
    ["1 Baseline", "train_baseline.py, checkpoints/resnet18_fp32.pth"],
    ["2 PTQ", "ptq.py, calibration.py, quant_utils.py, checkpoints/resnet18_ptq_int8(.pth, _ts.pt)"],
    ["3 QAT", "qat.py, checkpoints/resnet18_qat_int8(.pth, _ts.pt)"],
    ["4 Unstructured pruning", "prune_unstructured.py, checkpoints/resnet18_unstructured_pruned.pth"],
    ["5 Structured pruning", "prune_structured.py, checkpoints/resnet18_structured_pruned.pth"],
    ["6 Inference pipeline", "inference.py (all five models)"],
    ["7 Benchmarking", "benchmark.py, results/*.csv"],
    ["8 ONNX", "export_onnx.py, onnx_inference.py, onnx_models/*.onnx, results/onnx_benchmark.csv"],
    ["9 Report", "report/Technical_Report.pdf / .docx"],
    ["10 README", "README.md, requirements.txt, run_all.sh"],
], columns=["Deliverable", "Files"])
table(dm, font=8.5)

# ----------------------------------------------------------------------------- appendix
doc.add_page_break()
H("Appendix - console output of each step")
for name, cap in [("console_01_train_baseline.png", "A1 - baseline training (last epochs)"),
                  ("console_02_ptq.png", "A2 - PTQ"), ("console_03_qat.png", "A3 - QAT"),
                  ("console_04_unstructured.png", "A4 - unstructured pruning"),
                  ("console_05_structured.png", "A5 - structured pruning"),
                  ("console_08_export_onnx.png", "A6 - ONNX export"),
                  ("console_09_onnx_benchmark.png", "A7 - ONNX-Runtime benchmark")]:
    fig(name, cap, 17)

out = ROOT / "report" / "Technical_Report.docx"
doc.save(out)
print("wrote", out)
