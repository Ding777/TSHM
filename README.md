# Temporal–Spectral Hamiltonian Mixers for Efficient LongSequence Modeling



---

## TL;DR

This repository implements **Temporal–Spectral Hamiltonian Mixers (TSHM)** for long-sequence modeling. It contains code for forecasting and for audio classification with optional streaming (causal) inference. Use the `src/train.py` or `tshm_transformer70.py` entrypoints (examples below) to train and evaluate models on your data.

---

## Quick start — minimal steps

1. Clone the repo:

```bash
git clone <your-repo-url>
cd <your-repo-dir>
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install required packages (example):

```bash
pip install -r requirements.txt
# or:
pip install numpy pandas scikit-learn tqdm
pip install torch torchvision torchaudio   # choose appropriate wheel for your system
```

> Note: Install the correct wheel for PyTorch (CPU vs CUDA) according to your machine.

4. Run training / evaluation using one of the examples below.

---

## How to run — examples

### Audio classification (TSHM-based, streaming support)

```bash
# MFCC feature training
python src/train.py --data_dir ./speech_commands --mode mfcc --epochs 20 --batch_size 32 --d_model 48 --n_layers 3 --mode mfcc

# Raw audio (sequence length=16000) training
python src/train.py --data_dir ./speech_commands --mode raw --epochs 20 --batch_size 32 --d_model 48 --n_layers 3

# Streaming evaluation (requires causal model)
python src/train.py --data_dir ./speech_commands --mode mfcc --causal --streaming_eval
```

### Time-series forecasting (TSHMForecaster)

```bash
# Basic train with an ETT CSV (hourly)
python3 tshm_forecaste.py \
  --data_dir /path/to/ETTh1.csv \
  --model tshm \
  --epochs 15 \
  --dataset_class ETT_hour \
  --batch_size 32 \
  --d_model 256 \
  --n_layers 3 \
  --input_len 96 \
  --pred_len 168

# ETT-minute CSV
python3 tshm_forecaste.py \
  --data_dir /path/to/ETTm1.csv \
  --model tshm \
  --epochs 8 \
  --dataset_class ETT_minute \
  --batch_size 32 \
  --d_model 256 \
  --input_len 288 \
  --pred_len 48

# Single arbitrary CSV (ForecastCSV behavior — 80/10/10 split)
python3 tshm_forecaste.py \
  --data_dir /path/to/your_dataset.csv \
  --model tshm \
  --dataset_class ForecastCSV \
  --epochs 10 \
  --batch_size 16 \
  --input_len 192 \
  --pred_len 48

# Custom dataset folder (df_x.csv / df_y.csv or partitioned train/validation/test)
python3 tshm_forecaste.py \
  --data_dir /path/to/dataset_folder \
  --model tshm \
  --dataset_class Custom \
  --epochs 12 \
  --batch_size 16
```

To explicitly set device:

```bash
python3 tshm_forecaste.py ... --device cuda:0
```

---

## Outputs

* Best model checkpoint: `best_{model}_{dataset}.pth` (e.g., `best_tshm_ETTh1.pth`)
* Prediction CSV: `predictions_{model}_{dataset}_h{horizon}.csv`
* Console logs: per-epoch losses and metrics, diagnostics at the end

---

## Files / Structure (high level)

* `src/` — training & model code (audio classification)

  * `src/train.py` — training/eval driver for audio classification
  * `src/tshm/models.py` — TSHMBlock, TSHMStack, TSHMEncoder, TSHMClassifier (streaming)
  * `src/tshm/data.py` — dataset loaders and helpers
* `tshm_forecaste.py` — forecasting runner (ETT / M4 / ForecastCSV / Custom support)
* `requirements.txt` — recommended packages

---

## Requirements

* Python 3.8+ (3.9/3.10 recommended)
* Core Python packages: `numpy`, `pandas`, `scikit-learn`, `tqdm`
* PyTorch and `torchaudio` for audio work
* If using DataLoader wrappers, you may also need `torchvision` and `pyyaml` depending on your environment

Example CPU-only PyTorch install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

## Notes & Tips

* For streaming equivalence (batch vs step-by-step), build the model with `--causal`.
* If using pre-convolution layers, use `--causal` to ensure consistent causal conv behavior offline and in streaming.
* If your dataset is short relative to `input_len + pred_len`, you may get empty datasets — reduce `input_len` or `pred_len`.
* For debugging, use fewer epochs and smaller batch sizes.

---







