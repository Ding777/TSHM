
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

