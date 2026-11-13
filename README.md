# Assignment 03 — Edge Computing & Intelligent Systems

Model optimisation pipeline for a CIFAR-10 **ResNet-18**:

* **Q1** – Post-Training Static Quantisation (PTQ) and Quantisation-Aware Training (QAT) → INT8 models
* **Q2** – Unstructured (iterative magnitude) pruning and structured channel pruning with architecture reconstruction
* Unified inference pipeline, automated benchmark (CSV), and ONNX export + ONNX-Runtime CPU inference for all five models

The numbers, figures and discussion are in **`report/Technical_Report.pdf`** (editable version: `report/Technical_Report.docx`, regenerated from the CSVs by `report/build_report.py`).

---

## 1. Requirements

| | |
|---|---|
| Python | 3.9 – 3.12 |
| Packages | `pip install -r requirements.txt` (torch ≥ 2.2, torchvision, numpy, pandas, matplotlib, pillow, onnx, onnxruntime, tqdm) |
| Hardware | any CPU; an Apple-silicon GPU (MPS) or CUDA GPU is used automatically for FP32 training / pruning fine-tuning |
| Data | CIFAR-10 is downloaded automatically into `data/` on first run (~163 MB) |

The scripts were developed and run on an **Apple M4 (macOS, PyTorch 2.8, Python 3.9)**.
Quantised kernels: x86 machines use the **FBGEMM** backend, ARM machines (Apple silicon, phones, Raspberry Pi) use **QNNPACK**.
`common.select_quant_backend()` picks the right one automatically — no code change needed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. One-command reproduction

```bash
bash run_all.sh            # full pipeline, ~1.5–2 h on an M-series laptop (seeds fixed to 42)
bash run_all.sh --quick    # 1-epoch smoke test (minutes) – only checks that everything runs
```

Each step logs to `logs/<step>.log`.

## 3. Step-by-step

Run from the project root. Each step needs the outputs of the previous ones.

| # | Command | What it does | Output |
|---|---------|--------------|--------|
| 1 | `python train_baseline.py` | trains FP32 ResNet-18 (20 epochs, SGD + one-cycle LR, crop/flip augmentation) | `checkpoints/resnet18_fp32.pth` |
| 2 | `python ptq.py` | fuse Conv+BN+ReLU → set backend → calibrate on 2000 **training** images → convert to INT8 | `checkpoints/resnet18_ptq_int8.pth`, `…_ts.pt` |
| 3 | `python qat.py` | init from FP32 → fuse (QAT) → insert FakeQuantize → fine-tune → convert to INT8 | `checkpoints/resnet18_qat_int8.pth`, `…_ts.pt`, `…_qat_fakequant.pth` |
| 4 | `python prune_unstructured.py` | global L1 magnitude pruning to 90 % sparsity in 5 iterations, fine-tuning in between, masks made permanent | `checkpoints/resnet18_unstructured_pruned.pth` |
| 5 | `python prune_structured.py` | L1 channel pruning of the inner width of every residual block, rebuilds a physically smaller network, fine-tunes | `checkpoints/resnet18_structured_pruned.pth` |
| 6 | `python inference.py` | runs all five models on sample images (`--image my.png` for your own images) | console table |
| 7 | `python benchmark.py` | full 10 000-image test sweep, size / latency / throughput for all five models | `results/*.csv` |
| 8 | `python export_onnx.py` | exports FP32, PTQ-INT8, QAT-INT8, unstructured, structured to ONNX | `onnx_models/*.onnx` |
| 9 | `python onnx_inference.py` | CPU inference with ONNX-Runtime (accuracy + latency), or `--image my.png` | `results/onnx_benchmark.csv` |
| 10 | `python make_report_assets.py` | figures / console screenshots used in the report | `report/figures/*.png` |

Useful options: `python <script>.py --help`. Examples:

```bash
python ptq.py --calib-images 1000 --per-tensor       # different calibration size / per-tensor weights
python qat.py --epochs 5 --max-images 50000          # longer QAT (≈12 min per full epoch on CPU)
python prune_unstructured.py --sparsity 0.8 --iterations 4
python prune_structured.py --ratios 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
python inference.py --models fp32,qat,structured --image cat.png dog.png
python onnx_inference.py --models structured --image cat.png
```

## 4. Project layout

```
common.py              data pipeline, ResNet-18 (fusable + rebuildable), train/eval, size/latency helpers
train_baseline.py      Deliverable 1  – FP32 baseline training
quant_utils.py         qconfig / prepare / convert / TorchScript save+load helpers
calibration.py         Deliverable 2  – PTQ calibration pipeline
ptq.py                 Deliverable 2  – post-training static quantisation
qat.py                 Deliverable 3  – quantisation-aware training + fine-tuning
prune_utils.py         fine-tuning loop shared by both pruning scripts
prune_unstructured.py  Deliverable 4  – iterative magnitude pruning
prune_structured.py    Deliverable 5  – channel pruning + architecture reconstruction
inference.py           Deliverable 6  – inference pipeline for all five models
benchmark.py           Deliverable 7  – automated benchmark → CSV
export_onnx.py         Deliverable 8  – ONNX export
onnx_inference.py      Deliverable 8  – ONNX-Runtime CPU inference + benchmark
make_report_assets.py  figures for the report
run_all.sh             reproducible end-to-end pipeline
checkpoints/           model checkpoints (.pth state-dicts, .pt TorchScript for INT8 models)
onnx_models/           exported ONNX models
results/               CSV results (summary, per-image predictions, ONNX)
logs/                  training logs / console output of every step
report/                technical report (PDF/DOCX), figures, build_report.py
samples/               a few PNG images for trying inference.py --image
data/                  CIFAR-10 (downloaded automatically)
```

## 5. Checkpoint formats

| File | Content | Load with |
|------|---------|-----------|
| `resnet18_fp32.pth`, `resnet18_unstructured_pruned.pth`, `resnet18_structured_pruned.pth` | `{state_dict, mid_channels, meta}` – the structured model stores its reduced channel widths so the smaller architecture can be rebuilt | `common.load_float_model(path)` |
| `resnet18_ptq_int8_ts.pt`, `resnet18_qat_int8_ts.pt` | self-contained **TorchScript** INT8 models (**executable checkpoints**) | `torch.jit.load(path)` after `torch.backends.quantized.engine = "qnnpack"` (or `"fbgemm"` on x86) — or just `inference.load_model("ptq")` |
| `resnet18_ptq_int8.pth`, `resnet18_qat_int8.pth` | INT8 `state_dict`s (packed weights) | needs the same prepare/convert recipe (see `quant_utils.py`) |
| `resnet18_qat_fakequant.pth` | QAT model *before* INT8 conversion (fake-quant), used for the ONNX QDQ export | `quant_utils.build_qat_prepared` |

> **INT8 checkpoints are backend specific**: the packed weights are laid out for the quantisation backend that created them
> (QNNPACK here). Re-run `ptq.py` / `qat.py` (about 1 minute for PTQ) on an x86 machine to obtain FBGEMM versions.

## 6. Notes / honest caveats

* The CPU timing numbers are for **this laptop** (Apple M4, 1 thread by default — `--threads` to change); absolute values differ per machine.
* Unstructured pruning produces *zeros inside dense tensors*: accuracy and gzip size benefit, but latency does **not** improve without a sparse runtime. Structured pruning really shrinks the network (fewer parameters and MACs, lower latency).
* PyTorch's eager-mode INT8 operators cannot be exported to ONNX directly. The ONNX PTQ model is therefore produced by the ONNX-Runtime static quantiser from the FP32 ONNX graph (same calibration subset), and the ONNX QAT model carries the scales learned by QAT as QuantizeLinear/DequantizeLinear nodes. Details in the report.
* Test data is never used for calibration, training or model selection. The final-epoch model is the one that is saved.
