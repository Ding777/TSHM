# src/train.py
import argparse
import os
import math
import random
import time
from pathlib import Path
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tshm.models import TSHMClassifier
from tshm.data import FolderSpeechCommandsDataset, RemapLabelsDataset, collate_batch, mixup_data, SC10_CLASSES

try:
    from torchaudio.transforms import MFCC, Resample
except Exception:
    MFCC = None
    Resample = None

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct, total

def train_epoch(model, loader, optim, device, criterion, mixup_alpha=0.0, clip_grad=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for step, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        if y.dtype != torch.long:
            y = y.long()
        if y.numel() == 0:
            continue
        if y.min().item() < 0 or y.max().item() >= model.head[-1].out_features:
            print(f"[label-warning] batch {step}: label range out-of-bounds -> min {int(y.min())}, max {int(y.max())}, n_classes {model.head[-1].out_features}")
            y = y.clamp(0, model.head[-1].out_features - 1).to(device)

        if mixup_alpha and mixup_alpha > 0.0:
            x_mix, y_a, y_b, lam = mixup_data(x, y, mixup_alpha)
            logits = model(x_mix)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        else:
            logits = model(x)
            loss = criterion(logits, y)

        optim.zero_grad()
        loss.backward()
        if clip_grad and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optim.step()

        running_loss += float(loss.item()) * x.size(0)
        c, t = compute_accuracy(logits, y if mixup_alpha == 0.0 else y)
        correct += c
        total += t
    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def eval_epoch(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if y.dtype != torch.long:
                y = y.long()
            if y.numel() == 0:
                continue
            if y.min().item() < 0 or y.max().item() >= model.head[-1].out_features:
                y = y.clamp(0, model.head[-1].out_features - 1).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += float(loss.item()) * x.size(0)
            c, t = compute_accuracy(logits, y)
            correct += c
            total += t
    avg_loss = running_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./speech_commands", help="SpeechCommands folder (or root) to use or create")
    parser.add_argument("--mode", type=str, choices=["mfcc", "raw"], default="mfcc", help="Input features: 'mfcc' or 'raw'")
    parser.add_argument("--input_len", type=int, default=16000, help="Waveform length in samples (for raw) and MFCC reference (samples)")
    parser.add_argument("--n_mfcc", type=int, default=40, help="Number of MFCC coefficients")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--ff_hidden", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="If set, attempt torchaudio download of SpeechCommands when folder structure absent")

    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout in classifier head")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="Linear LR warmup epochs")
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine"], default="cosine", help="LR scheduler after warmup")
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Mixup alpha; 0 to disable")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for CrossEntropyLoss")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Clip grad norm (0 to disable)")

    parser.add_argument("--causal", action="store_true", help="Build encoder blocks in causal (streaming) mode")
    parser.add_argument("--streaming_eval", action="store_true", help="After training evaluate in streaming (step-wise) mode on test set (causal mode required)")

    # Augmentation controls
    parser.add_argument("--aug_per_clean", type=int, default=0, help="Number of augmented copies to create per clean training sample (0 disables)")
    parser.add_argument("--snr_min", type=float, default=-5.0, help="Minimum SNR (dB) for mixing background")
    parser.add_argument("--snr_max", type=float, default=10.0, help="Maximum SNR (dB) for mixing background")
    parser.add_argument("--bg_replications", type=int, default=0, help="IGNORED: background-only replications are NOT supported (to avoid adding noise as a label)")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    root = args.data_dir
    if not os.path.exists(root):
        raise FileNotFoundError(f"Data root {root} does not exist. Place downloaded folder there or pass correct path.")

    try:
        subdirs = [p for p in Path(root).iterdir() if p.is_dir()]
        has_class_subdirs = False
        for s in subdirs:
            if s.name == "_background_noise_":
                continue
            if s.name not in SC10_CLASSES:
                continue
            for ext in (".wav", ".flac", ".mp3"):
                if list(s.glob(f"*{ext}")):
                    has_class_subdirs = True
                    break
            if has_class_subdirs:
                break

        if has_class_subdirs:
            print("[loader] Detected class subfolders. Checking for validation/testing lists.")
            root_path = Path(root)
            valid_list_path = root_path / "validation_list.txt"
            test_list_path = root_path / "testing_list.txt"

            exts = (".wav", ".flac", ".mp3")
            all_files = []
            for sub in sorted(Path(root).iterdir()):
                if not sub.is_dir():
                    continue
                if sub.name == "_background_noise_":
                    continue
                if sub.name not in SC10_CLASSES:
                    continue
                for ext in exts:
                    for f in sub.glob(f"*{ext}"):
                        rel = os.path.relpath(f, root).replace("\\", "/")
                        all_files.append((str(f), sub.name, rel))
            if len(all_files) == 0:
                raise RuntimeError("No audio files found under subfolders.")

            val_set = set()
            test_set = set()
            if valid_list_path.exists():
                with open(valid_list_path, "r") as f:
                    for line in f:
                        p = line.strip().replace("\\", "/")
                        if p:
                            val_set.add(p)
            if test_list_path.exists():
                with open(test_list_path, "r") as f:
                    for line in f:
                        p = line.strip().replace("\\", "/")
                        if p:
                            test_set.add(p)

            train_items = []
            val_items = []
            test_items = []
            for fp, lab, rel in all_files:
                if rel in test_set:
                    test_items.append((fp, lab))
                elif rel in val_set:
                    val_items.append((fp, lab))
                else:
                    train_items.append((fp, lab))

            print(f"[loader] Using lists: train {len(train_items)} val {len(val_items)} test {len(test_items)}")
            train_labels = sorted(list({lab for _, lab in train_items}))

            # background folder detection (only as augmentation source)
            bg_folder = root_path / "_background_noise_"
            bg_files = []
            if bg_folder.exists() and bg_folder.is_dir():
                for ext in exts:
                    bg_files.extend([str(x) for x in bg_folder.glob(f"*{ext}")])
                bg_files = sorted(bg_files)
                if bg_files:
                    print(f"[loader] Found {len(bg_files)} background files for augmentation.")
            else:
                bg_files = []

            # IMPORTANT: do NOT add any background/noise label to train_labels.
            canonical_label2idx = {lab: i for i, lab in enumerate(train_labels)}

            class _FromList(Dataset):
                def __init__(self, items, mode, input_len, sr, n_mfcc):
                    self.items = items
                    self.mode = mode
                    self.input_len = input_len
                    self.sr = sr
                    self.n_mfcc = n_mfcc
                    self.mfcc_transform = MFCC(sample_rate=sr, n_mfcc=n_mfcc, log_mels=True, melkwargs=MFCC_MEL_KWARGS) if self.mode == "mfcc" else None
                    self.resampler_cache = {}

                def __len__(self):
                    return len(self.items)

                def __getitem__(self, idx):
                    fp, lab = self.items[idx]
                    waveform, sr = torchaudio.load(fp)
                    if sr != self.sr:
                        if sr not in self.resampler_cache:
                            self.resampler_cache[sr] = Resample(orig_freq=sr, new_freq=self.sr)
                        waveform = self.resampler_cache[sr](waveform)
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    L = waveform.shape[1]
                    if L < self.input_len:
                        waveform = F.pad(waveform, (0, self.input_len - L))
                    elif L > self.input_len:
                        waveform = waveform[:, : self.input_len]
                    if self.mode == "raw":
                        x = waveform.squeeze(0).unsqueeze(-1).numpy().astype(np.float32)
                    else:
                        mf = MFCC(sample_rate=self.sr, n_mfcc=self.n_mfcc, log_mels=True, melkwargs=MFCC_MEL_KWARGS)(waveform) if MFCC is not None else None
                        if mf is None:
                            raise RuntimeError("torchaudio.MFCC not available.")
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
                    return torch.from_numpy(x), lab

            # If augmentation requested and we found bg_files, use AugmentedFromList.
            if args.aug_per_clean > 0:
                if bg_files:
                    train_ds_str = AugmentedFromList(train_items, args.mode, args.input_len, 16000, args.n_mfcc, bg_files=bg_files, aug_per_clean=args.aug_per_clean, snr_min=args.snr_min, snr_max=args.snr_max)
                else:
                    print("[loader] augmentation requested but no background files found; using unaugmented training dataset.")
                    train_ds_str = _FromList(train_items, args.mode, args.input_len, 16000, args.n_mfcc)
            else:
                train_ds_str = _FromList(train_items, args.mode, args.input_len, 16000, args.n_mfcc)

            # NOTE: bg_replications is deliberately ignored to avoid creating background-as-label.
            if args.bg_replications > 0:
                print("[warning] --bg_replications was provided but is ignored. Background-only samples are NOT added as a class.")

            val_ds_str = _FromList(val_items, args.mode, args.input_len, 16000, args.n_mfcc)
            test_ds_str = _FromList(test_items, args.mode, args.input_len, 16000, args.n_mfcc)

            train_ds = RemapLabelsDataset(train_ds_str, canonical_label2idx, underlying_idx2label=None, fallback_to_last=False)
            val_ds = RemapLabelsDataset(val_ds_str, canonical_label2idx, underlying_idx2label=None, fallback_to_last=True)
            test_ds = RemapLabelsDataset(test_ds_str, canonical_label2idx, underlying_idx2label=None, fallback_to_last=True)
            classes_list = train_labels
        else:
            # fallback: attempt to use torchaudio SPEECHCOMMANDS or error
            if torchaudio is not None:
                try:
                    inner = SPEECHCOMMANDS(root, subset=None, download=False)
                except Exception:
                    raise RuntimeError("No subfolder layout and torchaudio SPEECHCOMMANDS not accessible.")
                walker = getattr(inner, "_walker", None)
                if walker is None:
                    raise RuntimeError("Cannot parse SPEECHCOMMANDS walker.")
                items = []
                for w in walker:
                    label = Path(w).parent.name
                    if label in SC10_CLASSES:
                        items.append(w)
                if len(items) == 0:
                    raise RuntimeError("No SC10 classes found in SPEECHCOMMANDS")
                indices = list(range(len(items)))
                random.shuffle(indices)
                N = len(indices)
                n_train = int(0.8 * N)
                n_val = int(0.1 * N)
                train_idx = indices[:n_train]
                val_idx = indices[n_train:n_train + n_val]
                test_idx = indices[n_train + n_val:]

                class _IdxSimple(Dataset):
                    def __init__(self, inner, idxs, mode, input_len, n_mfcc):
                        self.inner = inner
                        self.idxs = idxs
                        self.mode = mode
                        self.input_len = input_len
                        self.n_mfcc = n_mfcc

                    def __len__(self):
                        return len(self.idxs)

                    def __getitem__(self, i):
                        w = self.inner._walker[self.idxs[i]]
                        fp = str(Path(self.inner._path) / w)
                        waveform, sr = torchaudio.load(fp)
                        if sr != 16000:
                            waveform = Resample(orig_freq=sr, new_freq=16000)(waveform)
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                        L = waveform.shape[1]
                        if L < self.input_len:
                            waveform = F.pad(waveform, (0, self.input_len - L))
                        elif L > self.input_len:
                            waveform = waveform[:, :self.input_len]
                        if self.mode == "raw":
                            x = waveform.squeeze(0).unsqueeze(-1).numpy().astype(np.float32)
                        else:
                            mf = MFCC(sample_rate=16000, n_mfcc=args.n_mfcc, log_mels=True, melkwargs=MFCC_MEL_KWARGS)(waveform)
                            if mf.ndim == 3:
                                mf = mf.squeeze(0)
                            mf = mf.transpose(0, 1)
                            x = mf.numpy().astype(np.float32)
                            T = x.shape[0]
                            max_frames = 161
                            if T < max_frames:
                                x = np.pad(x, ((0, max_frames - T), (0, 0)), mode="constant")
                            elif T > max_frames:
                                x = x[:max_frames, :]
                        label = Path(w).parent.name
                        return torch.from_numpy(x), label

                labels_all = set()
                for w in walker:
                    lab = Path(w).parent.name
                    if lab in SC10_CLASSES:
                        labels_all.add(lab)
                train_labels = sorted(list(labels_all))
                canonical_label2idx = {lab: i for i, lab in enumerate(train_labels)}

                train_raw = _IdxSimple(inner, train_idx, args.mode, args.input_len, args.n_mfcc)
                val_raw = _IdxSimple(inner, val_idx, args.mode, args.input_len, args.n_mfcc)
                test_raw = _IdxSimple(inner, test_idx, args.mode, args.input_len, args.n_mfcc)

                train_ds = RemapLabelsDataset(train_raw, canonical_label2idx, underlying_idx2label=None, fallback_to_last=False)
                val_ds = RemapLabelsDataset(val_raw, canonical_label2idx, underlying_idx2label=None, fallback_to_last=True)
                test_ds = RemapLabelsDataset(test_raw, canonical_label2idx, underlying_idx2label=None, fallback_to_last=True)
                classes_list = train_labels
            else:
                raise RuntimeError("No class subfolders and torchaudio not available to parse SpeechCommands.")

    except Exception as e:
        print("Error while building datasets:", e)
        traceback.print_exc()
        raise

    n_classes = len(classes_list)
    print(f"[main] canonical number of classes (from training set): {n_classes}")
    print(f"[main] sample classes (first 30): {classes_list[:30]}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=2, pin_memory=True)

    sample_x, sample_y = train_ds[0]
    input_dim = sample_x.shape[1]
    print(f"[info] feature dim: {input_dim}")

    model = TSHMClassifier(input_dim=input_dim, n_classes=n_classes, d_model=args.d_model, n_layers=args.n_layers, r=args.r, K=args.K, ff_hidden=args.ff_hidden, use_pos=not args.causal, dropout=args.dropout, causal=args.causal)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Parameters:", total_params)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing if args.label_smoothing > 0 else 0.0)
    except TypeError:
        if args.label_smoothing > 0:
            print("[warning] label_smoothing requested but CrossEntropyLoss doesn't support it in this PyTorch; ignoring.")
        criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    def lr_for_epoch(epoch):
        base = args.lr
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return base * float(epoch + 1) / float(max(1, args.warmup_epochs))
        if args.scheduler == "cosine" and args.epochs > args.warmup_epochs:
            t = epoch - args.warmup_epochs
            T = max(1, args.epochs - args.warmup_epochs)
            return base * 0.5 * (1.0 + math.cos(math.pi * float(t) / float(T)))
        return base

    for epoch in range(1, args.epochs + 1):
        lr_now = lr_for_epoch(epoch - 1)
        for g in optim.param_groups:
            g["lr"] = lr_now
        print(f"[epoch] {epoch}/{args.epochs} lr={lr_now:.6e}")
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optim, device, criterion, mixup_alpha=args.mixup_alpha, clip_grad=args.clip_grad if args.clip_grad > 0 else None)
        val_loss, val_acc = eval_epoch(model, val_loader, device, criterion)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.1f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            torch.save(best_state, f"best_tshm_{args.mode}.pth")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[info] Best val acc: {best_val_acc:.4f}")

    test_loss, test_acc = eval_epoch(model, test_loader, device, criterion)
    print(f"[result] Test loss={test_loss:.4f} acc={test_acc:.4f}")

    # Streaming evaluation block — replaced to use classifier forward_step properly
    if args.streaming_eval:
        if not args.causal:
            print("[streaming-eval] --streaming_eval requested but model built in non-causal mode. Rebuilding causal model for streaming evaluation.")
            causal_model = TSHMClassifier(input_dim=input_dim, n_classes=n_classes, d_model=args.d_model, n_layers=args.n_layers, r=args.r, K=args.K, ff_hidden=args.ff_hidden, use_pos=False, dropout=args.dropout, causal=True).to(device)
            try:
                causal_model.load_state_dict(model.state_dict(), strict=False)
                print("[streaming-eval] loaded weights into causal model (non-strict).")
                model = causal_model
            except Exception as e:
                print("[streaming-eval] Warning: could not load weights exactly into causal model:", e)
        model.eval()
        print("[streaming-eval] Running streaming evaluation on test set (step-wise). This may be slower than batched eval.")
        total = 0
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                B, L, feat = x_batch.shape
                # initialize classifier-level streaming state (handles conv-buffer if needed)
                states = model.init_stream_state(batch_size=B, device=device)
                # collect per-step hidden states (encoder outputs after residual mixing) to pool later
                h_list = []
                for t in range(L):
                    x_t = x_batch[:, t, :]  # (B, input_dim)
                    logits_t, states, h_t = model.forward_step(x_t, states)  # h_t is (B, d_model)
                    h_list.append(h_t.unsqueeze(1))
                out_seq = torch.cat(h_list, dim=1)  # (B, L, d_model)
                #pooled = out_seq.mean(dim=1)
                pooled = out_seq.max(dim=1)[0]
                logits = model.head(pooled)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        acc_stream = correct / total if total > 0 else 0.0
        print(f"[streaming-eval] streaming test accuracy: {acc_stream:.4f}")

    model.eval()
    examples = 8
    printed = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_np = y.numpy()
            for i in range(x.size(0)):
                pred_label = classes_list[preds[i]] if 0 <= preds[i] < len(classes_list) else str(preds[i])
                true_label = classes_list[y_np[i]] if 0 <= y_np[i] < len(classes_list) else str(y_np[i])
                print(f"Example {printed+1}: pred='{pred_label}'  true='{true_label}'")
                printed += 1
                if printed >= examples:
                    break
            if printed >= examples:
                break


if __name__ == "__main__":
    main()
