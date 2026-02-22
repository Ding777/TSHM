This is the code of Temporal–Spectral Hamiltonian Mixers for Efficient LongSequence Modeling

Abstract We present Temporal Spectral Hamiltonian Mixer, a lightweight and memory-efficient architecture for modelinglong-range dependencies in sequential data. Inspired by Hamiltonian dynamics, TSHM introduces a structuredmodulethatmaintains stability and long-term coherence while remaining simple to implement, efficient to train, and can handletentooverhundred thousand of sequence length without memory overflow. The design supports both parallelized trainingandconstanttime streaming inference O (1), making it suitable for real-time and low-latency applications. Compared with Transformersandstructured state-space models(SSM) such as S4, TSHM achieves a favorable balance between expressivity, computational cost,and engineering simplicity and are naturally bidirectional or sequential in all direction. It avoids specialized kernels, requiresonly dense linear and pointwise operations, and operates with minimal memory overhead. We provide a mathematicalformulation, interpret the model through a Hamiltonian lens, analyze its computational complexity, and outline a reproducibleexperimental plan across diverse benchmarks in speech recognition, text classification, time-series forecasting, andsequentialimage classification. Across benchmarks — Long Range Area(LRA), sMNIST, CIFAR (1-D), long horizon forecast datasetandGoogle Speech Commands, TSHM consistently outperforms Transformer variants, it is second or sometime competitivewithSSM (S4), and shows largest gains on time-series forecasting. Notably, TSHM processes raw 16,000-sample speechsequencesachieving 91% accuracy, LRA Path-X (16000 sequence length) achieving 80.31% while standard Transformer andRecurrentNeural Network (RNN) baselines fail under the same conditions because of memory overflow or efficiency. Finally,todemonstrate TSHM’s real-world practicality, we deployed it as firmware on an ESP32-S3 voice-control device. Themodel runsat 1.1–1.4 ms per frame and completes 10 head-only Stochastic Gradient Descent (SGD) updates in 1.2 ms, enablingaccurate,privacy-preserving on-device learning without cloud 
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

