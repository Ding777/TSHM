# src/tshm/data.py
import os
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
try:
    import torchaudio
    from torchaudio.transforms import MFCC, Resample
    from torchaudio.datasets import SPEECHCOMMANDS
except Exception:
    torchaudio = None
    MFCC = None
    Resample = None
    SPEECHCOMMANDS = None

MFCC_MEL_KWARGS = {"n_fft": 512, "hop_length": 100, "n_mels": 64}
SC10_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

class FolderSpeechCommandsDataset(Dataset):
    def __init__(self, root, mode="mfcc", input_len=16000, sr=16000, n_mfcc=40, exts=(".wav", ".flac", ".mp3")):
        if torchaudio is None:
            raise RuntimeError("torchaudio required for folder loader.")
        self.root = Path(root)
        self.mode = mode
        self.input_len = int(input_len)
        self.sr = int(sr)
        self.n_mfcc = int(n_mfcc)
        self.exts = exts
        candidates = []
        for sub in sorted(self.root.iterdir()):
            if sub.is_dir() and sub.name == "_background_noise_":
                continue
            if sub.is_dir() and sub.name in SC10_CLASSES:
                found = False
                for ext in self.exts:
                    if list(sub.glob(f"*{ext}")):
                        found = True
                        break
                if found:
                    candidates.append(sub)
        if len(candidates) == 0:
            files = []
            for ext in self.exts:
                files += list(self.root.glob(f"*{ext}"))
            if len(files) == 0:
                raise RuntimeError(f"No audio files or class subfolders found under {root}")
            self.labels = ["all"]
            self.label2idx = {"all": 0}
            self.items = [str(p) for p in files]
        else:
            labels = [p.name for p in candidates]
            labels = sorted(labels)
            self.labels = labels
            self.label2idx = {lab: i for i, lab in enumerate(self.labels)}
            items = []
            for lab in self.labels:
                p = self.root / lab
                for ext in self.exts:
                    items.extend([str(x) for x in p.glob(f"*{ext}")])
            if len(items) == 0:
                raise RuntimeError(f"No audio files found in detected class subfolders under {root}")
            self.items = sorted(items)
        self.mfcc_transform = MFCC(sample_rate=self.sr, n_mfcc=self.n_mfcc, log_mels=True, melkwargs=MFCC_MEL_KWARGS) if self.mode == "mfcc" else None
        self.resampler_cache = {}
        print(f"[folder-loader] root={root} classes={len(self.labels)} samples={len(self.items)} mode={self.mode}")

    def __len__(self):
        return len(self.items)

    def _load_file(self, filepath):
        waveform, sr = torchaudio.load(filepath)
        return waveform, sr

    def _process_waveform(self, waveform, sample_rate):
        if sample_rate != self.sr:
            if sample_rate not in self.resampler_cache:
                self.resampler_cache[sample_rate] = Resample(orig_freq=sample_rate, new_freq=self.sr)
            waveform = self.resampler_cache[sample_rate](waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        L = waveform.shape[1]
        if L < self.input_len:
            pad = self.input_len - L
            waveform = F.pad(waveform, (0, pad))
        elif L > self.input_len:
            waveform = waveform[:, : self.input_len]
        return waveform

    def __getitem__(self, idx):
        fp = self.items[idx]
        label_name = None
        p = Path(fp)
        if p.parent != self.root:
            if p.parent.name in self.label2idx:
                label_name = p.parent.name
        if label_name is None:
            label_name = self.labels[0]
        waveform, sr = self._load_file(fp)
        waveform = self._process_waveform(waveform, sr)
        if self.mode == "raw":
            x = waveform.squeeze(0).unsqueeze(-1).numpy().astype(np.float32)
        else:
            with torch.no_grad():
                mf = MFCC(sample_rate=self.sr, n_mfcc=self.n_mfcc, log_mels=True, melkwargs=MFCC_MEL_KWARGS)(waveform) if self.mfcc_transform is None else self.mfcc_transform(waveform)
                if mf.ndim == 3:
                    mf = mf.squeeze(0)
            mf = mf.transpose(0, 1)
            x = mf.numpy().astype(np.float32)
            T = x.shape[0]
            max_frames = 161
            if T < max_frames:
                pad_t = max_frames - T
                x = np.pad(x, ((0, pad_t), (0, 0)), mode="constant")
            elif T > max_frames:
                x = x[:max_frames, :]
        y = self.label2idx.get(label_name, 0)
        return torch.from_numpy(x), int(y)


class RemapLabelsDataset(Dataset):
    def __init__(self, ds, canonical_label2idx, underlying_idx2label=None, fallback_to_last=True):
        self.ds = ds
        self.canonical = canonical_label2idx
        self.under_idx2lab = underlying_idx2label
        self.fallback = fallback_to_last
        self.last_idx = max(canonical_label2idx.values()) if canonical_label2idx else 0

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        if isinstance(y, torch.Tensor):
            try:
                y = int(y.item())
            except Exception:
                try:
                    y = y.tolist()
                except Exception:
                    y = int(y)
        if isinstance(y, int) and self.under_idx2lab is not None:
            label_str = self.under_idx2lab.get(int(y), None)
            if label_str is None:
                mapped = self.last_idx if self.fallback else 0
                return x, int(mapped)
            mapped = self.canonical.get(label_str, None)
            if mapped is None:
                mapped = self.last_idx if self.fallback else 0
            return x, int(mapped)
        if isinstance(y, str):
            mapped = self.canonical.get(y, None)
            if mapped is None:
                mapped = self.last_idx if self.fallback else 0
            return x, int(mapped)
        if isinstance(y, int):
            if 0 <= y <= self.last_idx:
                return x, int(y)
            else:
                mapped = self.last_idx if self.fallback else 0
                return x, int(mapped)
        return x, int(self.last_idx)


# Additional augmentation & helpers can be added here if needed (e.g., AugmentedFromList).
def collate_batch(batch):
    xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    maxlen = max(lengths)
    feat_dim = xs[0].shape[1]
    batch_x = torch.zeros((len(xs), maxlen, feat_dim), dtype=torch.float32)
    for i, x in enumerate(xs):
        L = x.shape[0]
        batch_x[i, :L, :] = x
    batch_y = torch.tensor(ys, dtype=torch.long)
    return batch_x, batch_y


def mixup_data(x, y, alpha):
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = float(lam)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    x2 = x[index]
    y2 = y[index]
    x_mix = lam * x + (1 - lam) * x2
    return x_mix, y, y2, lam
