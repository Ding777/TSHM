# Temporal–Spectral Hamiltonian Mixers for Efficient LongSequence Modeling

A concise, well-formatted README for GitHub. Below you’ll find a short polished overview, clear “How to run” instructions, and — per your request — the original text you provided included verbatim (unchanged).

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
python3 tshm_transformer70.py \
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
python3 tshm_transformer70.py \
  --data_dir /path/to/ETTm1.csv \
  --model tshm \
  --epochs 8 \
  --dataset_class ETT_minute \
  --batch_size 32 \
  --d_model 256 \
  --input_len 288 \
  --pred_len 48

# Single arbitrary CSV (ForecastCSV behavior — 80/10/10 split)
python3 tshm_transformer70.py \
  --data_dir /path/to/your_dataset.csv \
  --model tshm \
  --dataset_class ForecastCSV \
  --epochs 10 \
  --batch_size 16 \
  --input_len 192 \
  --pred_len 48

# Custom dataset folder (df_x.csv / df_y.csv or partitioned train/validation/test)
python3 tshm_transformer70.py \
  --data_dir /path/to/dataset_folder \
  --model tshm \
  --dataset_class Custom \
  --epochs 12 \
  --batch_size 16
```

To explicitly set device:

```bash
python3 tshm_transformer70.py ... --device cuda:0
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
* `tshm_transformer70.py` — forecasting runner (ETT / M4 / ForecastCSV / Custom support)
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

## Original content (verbatim; unchanged)

Below is the exact text you supplied, included verbatim and **not modified**.

````
WRITE README FPR GITHUB, HOW TO RUN

NOT MODFY THE TEXT, WRITE WELL , TITLE MUST BE WELL # Temporal–Spectral Hamiltonian Mixers for Efficient LongSequence Modeling

This is the code of Temporal–Spectral Hamiltonian Mixers for Efficient LongSequence Modeling

# TSHM Audio Classification

Project layout for TSHM-based audio classification with streaming (causal) support.

Run training/eval via `python src/train.py` (see usage below).

## Quickstart

1. Create and activate virtualenv:
```bash
bash setup.sh
source .venv/bin/activate

Edit or download SpeechCommands dataset into ./speech_commands (don't commit audio to git).

Example training:
MFCC feature train:
python src/train.py --data_dir ./speech_commands --mode mfcc --epochs 20 --batch_size 32 --d_model 48 --n_layers 3 --mode mfcc
raw audio(sequence length=16000) train:
python src/train.py --data_dir ./speech_commands --mode mfcc --epochs 20 --batch_size 32 --d_model 48 --n_layers 3 --mode raw


Streaming evaluation (requires causal model):

python src/train.py --data_dir ./speech_commands --mode mfcc --causal --streaming_eval
Files

src/tshm/models.py — TSHMBlock, TSHMStack, TSHMEncoder, TSHMClassifier (streaming)

src/tshm/data.py — dataset loaders and helpers (MFCC support via torchaudio)

src/train.py — training and evaluation driver

Notes

For exact batch-vs-stream equivalence:

Build model with --causal if you need causal streaming equivalence.

If using pre-convolution, use --causal so conv is applied causally both offline and streaming.

Run in model.eval() for evaluation.

2) `requirements.txt`
```text
numpy
torch>=1.11.0
torchaudio>=0.11.0
tqdm


# TSHM FORECASTE

Overview

This repository contains an implementation of a time-series forecaster (TSHM) with support for multiple dataset formats (ETT, M4, single CSV, partitioned folders, ForecastCSV-style sequences) and both batch and streaming (causal) modes.

Key features

Train/evaluate a TSHMForecaster model (configurable d_model, n_layers, r, K, etc.).

Multiple dataset loaders and an 80/10/10 single-CSV split helper.

Save best model checkpoint and per-sample prediction CSV export.

Streaming-friendly API (init_state, forward_step) for causal inference.

Diagnostics / per-horizon error reporting.

Requirements

Python 3.8+ (3.9/3.10 recommended)

Install the usual scientific stack: numpy, pandas, scikit-learn

Install PyTorch (CPU or CUDA) — for example:

# CPU-only (example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

Or, install CUDA-enabled wheel from PyTorch (choose the correct CUDA version for your machine).

Other required packages:

pip install numpy pandas scikit-learn tqdm

If you use the included DataLoader/wrappers, you may also need torchvision and pyyaml depending on environment.

Quick install
git clone <your-repo-url>
cd <your-repo-dir>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # if provided, else install above packages manually

(If you don't have a requirements.txt, install the packages listed in Requirements.)

How to run (examples)

Basic train with an ETT CSV (hourly)

python3 tshm_transformer70.py \
  --data_dir /path/to/ETTh1.csv \
  --model tshm \
  --epochs 15 \
  --dataset_class ETT_hour \
  --batch_size 32 \
  --d_model 256 \
  --n_layers 3 \
  --input_len 96 \
  --pred_len 168

Train using an ETT-minute CSV

python3 tshm_transformer70.py \
  --data_dir /path/to/ETTm1.csv \
  --model tshm \
  --epochs 8 \
  --dataset_class ETT_minute \
  --batch_size 32 \
  --d_model 256 \
  --input_len 288 \
  --pred_len 48

Train on a single arbitrary CSV (auto 80/10/10 split, ForecastCSV behavior)

python3 tshm_transformer70.py \
  --data_dir /path/to/your_dataset.csv \
  --model tshm \
  --dataset_class ForecastCSV \
  --epochs 10 \
  --batch_size 16 \
  --input_len 192 \
  --pred_len 48

Train on custom dataset folder (df_x.csv / df_y.csv or partitioned train/validation/test folders)

python3 tshm_transformer70.py \
  --data_dir /path/to/dataset_folder \
  --model tshm \
  --dataset_class Custom \
  --epochs 12 \
  --batch_size 16

Specify device (CUDA) explicitly

python3 tshm_transformer70.py ... --device cuda:0
Common example commands (from the repo)

Minute weather example:

python3 tshm_transformer70.py --data_dir /workspace/EETok/weather/weather.csv --model tshm --epochs 7 --dataset_class ETT_minute

Hourly ETT example:

python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTh2.csv --model tshm --epochs 15 --dataset_class ETT_hour

Single CSV custom split:

python3 tshm_transformer70.py --data_dir /workspace/time_serie --model tshm --epochs 2 --dataset_class custom --batch_size 8 --d_model 128 --ff_hidden 128 --n_layers 4 --r 64 --input_len 86 --pred_len 48

Use these to copy/paste or adapt.
````

---

## Want more?

If you’d like, I can:

* produce a ready-to-drop `requirements.txt` (based on the text above),
* generate a one-file `run.sh` with common experiment commands,
* clean and deduplicate the script (remove duplicated `PositionalEncoding` / encoder),
* or create a short CONTRIBUTING.md and example `LICENSE` (MIT).

Which would you like next?





