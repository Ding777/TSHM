#!/usr/bin/env python3
"""
tshm_forecast_ett_fixed_with_diag.py

Same runner as before but with a diagnostic helper suite added:
 - predict_on_loader
 - diag_print_scaler_stats
 - print_examples
 - per_horizon_errors
 - diagnose

Dataset preparation classes are included so you can call them as a library.
"""
import argparse
import os
import math
import random
import time
from pathlib import Path
import csv
import sys
import re
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# dataset/util imports used by dataset preparation classes
from datasets import load_dataset
from huggingface_hub import hf_hub_download
# utils.timefeatures and utils.augmentation are expected in your repo (kept as before)
try:
    from utils.timefeatures import time_features
except Exception:
    time_features = None
try:
    from utils.augmentation import run_augmentation_single
except Exception:
    run_augmentation_single = None

warnings.filterwarnings('ignore')

# HuggingFace repo used by some dataset loaders
HUGGINGFACE_REPO = "thuml/Time-Series-Library"
import sys
sys.path.append("/workspace/Time-Series-Library")

# --------------------------
# Dataset / data-prep classes (ETT, Custom, M4, segmentation loaders, UEA loader, ...)
# --------------------------

class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = int(size[0])
            self.label_len = int(size[1])
            self.pred_len = int(size[2])
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            split = "train" if "train" in ds else list(ds.keys())[0]
            df_raw = ds[split].to_pandas()

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ('M', 'MS'):
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # time stamp handling
        if 'date' in df_raw.columns:
            df_stamp = df_raw[['date']][border1:border2].copy()
            df_stamp['date'] = pd.to_datetime(df_stamp['date'])
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp['date'].dt.month
                df_stamp['day'] = df_stamp['date'].dt.day
                df_stamp['weekday'] = df_stamp['date'].dt.weekday
                df_stamp['hour'] = df_stamp['date'].dt.hour
                data_stamp = df_stamp.drop(columns=['date']).values
            else:
                if time_features is not None:
                    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                    data_stamp = data_stamp.transpose(1, 0)
                else:
                    data_stamp = np.zeros((border2 - border1, 4), dtype=np.float32)
        else:
            data_stamp = np.zeros((border2 - border1, 4), dtype=np.float32)

        self.data_x = data[border1:border2].astype(np.float32)
        self.data_y = data[border1:border2].astype(np.float32)
        self.data_stamp = data_stamp.astype(np.float32)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x.astype(np.float32), seq_y.astype(np.float32), seq_x_mark.astype(np.float32), seq_y_mark.astype(np.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_scaler(self):
        return self.scaler


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = int(size[0])
            self.label_len = int(size[1])
            self.pred_len = int(size[2])
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            split = "train" if "train" in ds else list(ds.keys())[0]
            df_raw = ds[split].to_pandas()

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ('M', 'MS'):
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if 'date' not in df_raw.columns:
            raise RuntimeError("Minute ETT expects a 'date' column for time features.")
        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            df_stamp['minute'] = df_stamp['date'].dt.minute
            df_stamp['minute'] = df_stamp['minute'].map(lambda x: x // 15)
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            if time_features is not None:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
            else:
                data_stamp = np.zeros((border2 - border1, 6), dtype=np.float32)
        else:
            data_stamp = np.zeros((border2 - border1, 6), dtype=np.float32)

        self.data_x = data[border1:border2].astype(np.float32)
        self.data_y = data[border1:border2].astype(np.float32)

        if self.set_type == 0 and getattr(self.args, "augmentation_ratio", 0) > 0 and (run_augmentation_single is not None):
            try:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            except Exception:
                pass

        self.data_stamp = data_stamp.astype(np.float32)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x.astype(np.float32), seq_y.astype(np.float32), seq_x_mark.astype(np.float32), seq_y_mark.astype(np.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_scaler(self):
        return self.scaler


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = int(size[0])
            self.label_len = int(size[1])
            self.pred_len = int(size[2])
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            df_raw = ds[split_name].to_pandas()

        if 'date' not in df_raw.columns:
            raise RuntimeError("Custom loader expects 'date' column in CSV.")
        if self.target not in df_raw.columns:
            raise RuntimeError(f"Custom loader expected target column '{self.target}' in CSV.")

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ('M', 'MS'):
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        else:
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1 and time_features is not None:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            data_stamp = np.zeros((border2 - border1, 4), dtype=np.float32)

        self.data_x = data[border1:border2].astype(np.float32)
        self.data_y = data[border1:border2].astype(np.float32)

        if self.set_type == 0 and getattr(self.args, "augmentation_ratio", 0) > 0 and (run_augmentation_single is not None):
            try:
                self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            except Exception:
                pass

        self.data_stamp = data_stamp.astype(np.float32)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x.astype(np.float32), seq_y.astype(np.float32), seq_x_mark.astype(np.float32), seq_y_mark.astype(np.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_scaler(self):
        return self.scaler


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        if size is None:
            raise RuntimeError("Dataset_M4 requires size tuple (seq_len,label_len,pred_len)")
        self.seq_len = int(size[0])
        self.label_len = int(size[1])
        self.pred_len = int(size[2])

        self.seasonal_patterns = seasonal_patterns
        # history size fallback
        try:
            from data_provider.m4 import M4Meta
            self.history_size = M4Meta.history_size[seasonal_patterns]
        except Exception:
            self.history_size = self.pred_len * 10
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        try:
            from data_provider.m4 import M4Dataset
            if self.flag == 'train':
                dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
            else:
                dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
            training_values = np.array(
                [v[~np.isnan(v)] for v in
                 dataset.values[dataset.groups == self.seasonal_patterns]])
            self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
            self.timeseries = [ts for ts in training_values]
        except Exception as e:
            raise RuntimeError("M4Dataset dependency error: " + str(e))

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1), dtype=np.float32)
        insample_mask = np.zeros((self.seq_len, 1), dtype=np.float32)
        outsample = np.zeros((self.pred_len + self.label_len, 1), dtype=np.float32)
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1), dtype=np.float32)

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        # M4 loader might not use scaler; keep compatibility
        try:
            return self.scaler.inverse_transform(data)
        except Exception:
            return data

class ForecastCSVSequence(Dataset):
    def __init__(self, data_dir, input_len=96, pred_len=24, stride=1, x_columns=None, y_columns=None, normalize=True, scaler=None):
        self.data_dir = Path(data_dir)
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.normalize = normalize
        x_path = self.data_dir / "df_x.csv"
        y_path = self.data_dir / "df_y.csv"
        self.single_csv = None
        if x_path.exists() and y_path.exists():
            self.X_df = pd.read_csv(x_path)
            self.Y_df = pd.read_csv(y_path)
        else:
            candidates = list(self.data_dir.glob("*.csv"))
            if len(candidates) == 0:
                raise FileNotFoundError(f"No CSV files found in {data_dir}. Expected df_x.csv & df_y.csv or a single CSV.")
            candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
            self.single_csv = candidates[0]
            df = pd.read_csv(self.single_csv)
            if x_columns is not None and y_columns is not None:
                self.X_df = df[x_columns].copy()
                self.Y_df = df[y_columns].copy()
            else:
                x_cols = [c for c in df.columns if c.startswith("x_")]
                y_cols = [c for c in df.columns if c.startswith("y_")]
                if not x_cols:
                    possible_y = [c for c in df.columns if ("target" in c.lower()) or (c.startswith("y_")) or (c.lower() == "y") or (c.lower() == "value")]
                    if possible_y:
                        y_cols = possible_y
                        x_cols = [c for c in df.columns if c not in y_cols]
                    else:
                        y_cols = [df.columns[-1]]
                        x_cols = list(df.columns[:-1])
                self.X_df = df[x_cols].copy()
                self.Y_df = df[y_cols].copy()

        self.X_df = self.X_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
        self.Y_df = self.Y_df.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
        self.X = self.X_df.values.astype(np.float32)
        self.Y = self.Y_df.values.astype(np.float32)
        if self.X.shape[0] != self.Y.shape[0]:
            n = min(self.X.shape[0], self.Y.shape[0])
            print(f"[loader] Warning: X and Y lengths differ ({self.X.shape[0]} vs {self.Y.shape[0]}). Trimming to {n}.")
            self.X = self.X[:n]
            self.Y = self.Y[:n]

        if scaler is None and self.normalize:
            self.x_mean = self.X.mean(axis=0, keepdims=True)
            self.x_std = self.X.std(axis=0, keepdims=True) + 1e-6
            self.y_mean = self.Y.mean(axis=0, keepdims=True)
            self.y_std = self.Y.std(axis=0, keepdims=True) + 1e-6
        elif scaler is not None:
            self.x_mean = scaler['x_mean']
            self.x_std = scaler['x_std']
            self.y_mean = scaler['y_mean']
            self.y_std = scaler['y_std']
        else:
            self.x_mean = np.zeros((1, self.X.shape[1]), dtype=np.float32)
            self.x_std = np.ones((1, self.X.shape[1]), dtype=np.float32)
            self.y_mean = np.zeros((1, self.Y.shape[1]), dtype=np.float32)
            self.y_std = np.ones((1, self.Y.shape[1]), dtype=np.float32)

        T = self.X.shape[0]
        max_start = T - (self.input_len + self.pred_len)
        if max_start < 0:
            self.starts = []
        else:
            self.starts = list(range(0, max_start + 1, self.stride))

    def get_scaler_dict(self):
        return {'x_mean': self.x_mean, 'x_std': self.x_std, 'y_mean': self.y_mean, 'y_std': self.y_std}

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        inp = self.X[s : s + self.input_len].astype(np.float32)
        tgt = self.Y[s + self.input_len : s + self.input_len + self.pred_len].astype(np.float32)
        if self.normalize:
            inp = (inp - self.x_mean) / self.x_std
            tgt = (tgt - self.y_mean) / self.y_std
        return torch.tensor(inp, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

# --------------------------
# TSHM model blocks (unchanged)
# --------------------------
import torch.nn.init as init
class TSHMBlockSimple(nn.Module):

    def __init__(self, d_model, r=32, K=8, ff_hidden=512, gate_kernel=3, per_channel_gate=False, gate_bias_init=-1.0, res_scale_init=0.2, causal=False):
        super().__init__()
        self.d = d_model
        self.causal = causal
        self.U = nn.Linear(d_model, r, bias=False)
        self.V = nn.Linear(d_model, r, bias=False)
        self.A = nn.Linear(r, K, bias=True)
        self.B = nn.Linear(r, K, bias=True)
        self.c = nn.Parameter(torch.zeros(K))
        out_ch = d_model if per_channel_gate else 1
        self.gate_kernel = gate_kernel
        self.gate_conv = nn.Conv1d(in_channels=d_model, out_channels=out_ch, kernel_size=gate_kernel, padding=0)
        nn.init.normal_(self.gate_conv.weight, std=0.02)
        nn.init.constant_(self.gate_conv.bias, gate_bias_init)
        self.res_scale = nn.Parameter(torch.tensor(res_scale_init))
        self.pre_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.LayerNorm(d_model),
                                 nn.Linear(d_model, ff_hidden),
                                 nn.GELU() if hasattr(nn, "GELU") else nn.Identity(),
                                 nn.Linear(ff_hidden, d_model))
        for lin in (self.U, self.V, self.A, self.B):
            if hasattr(lin, "weight"):
                nn.init.xavier_normal_(lin.weight)

    def forward(self, X):
        B, L, d = X.shape
        X_norm = self.pre_ln(X)
        phi = torch.tanh(X_norm)
        P = self.U(phi)
        S = torch.cumsum(P, dim=1)
        Q = self.V(phi)
        eA = self.A(S)
        eB = self.B(Q)
        e = eA + eB + self.c.view(1,1,-1)
        e = torch.tanh(e)
        w = 0.1
        G = (w * (0.1 ** 2)) * e

        A_w = self.A.weight
        M = torch.einsum('blk,kr->blr', G, A_w)

        if self.causal:
            M_prefix_cumsum = torch.cumsum(M, dim=1)
            term1_pre = torch.einsum('blr,rd->bld', M_prefix_cumsum, self.U.weight)
            N_all = torch.einsum('blk,kr->blr', G, self.B.weight)
            N_prefix_cumsum = torch.cumsum(N_all, dim=1)
            term2_pre = torch.einsum('blr,rd->bld', N_prefix_cumsum, self.V.weight)
            grad = term1_pre + term2_pre
        else:
            M_rev = torch.flip(M, dims=[1])
            M_rev_cumsum = torch.cumsum(M_rev, dim=1)
            M_suf = torch.flip(M_rev_cumsum, dims=[1])
            term1_pre = torch.einsum('blr,rd->bld', M_suf, self.U.weight)
            N_all = torch.einsum('blk,kr->blr', G, self.B.weight)
            term2_pre = torch.einsum('blr,rd->bld', N_all, self.V.weight)
            grad = term1_pre + term2_pre

        g_in = X_norm.permute(0,2,1).contiguous()
        if self.causal:
            pad = self.gate_kernel - 1
            g_in_padded = F.pad(g_in, (pad, 0)) if pad > 0 else g_in
            g_raw = self.gate_conv(g_in_padded).permute(0,2,1)
            g_raw = g_raw[:, -L:, :] if g_raw.shape[1] > L else g_raw
        else:
            pad_total = (self.gate_kernel - 1) // 2
            if pad_total > 0:
                g_in_padded = F.pad(g_in, (pad_total, pad_total))
                g_raw = self.gate_conv(g_in_padded).permute(0,2,1)
                g_raw = g_raw[:, :L, :]
            else:
                g_raw = self.gate_conv(g_in).permute(0,2,1)

        if g_raw.shape[-1] == 1:
            g = torch.sigmoid(g_raw).expand(-1,-1,d)
        else:
            g = torch.sigmoid(g_raw)
        eps = 1e-6
        g = eps + (1.0 - eps) * g
        X_next = X + (self.res_scale * g * grad)

        if isinstance(self.ffn[1], nn.Linear) and (hasattr(nn, "GELU") and isinstance(self.ffn[2], nn.GELU) or (not hasattr(nn, "GELU") and isinstance(self.ffn[2], nn.Identity))):
            X_next = X_next + self.ffn(X_next)
        else:
            if hasattr(F, "gelu"):
                tmp = F.gelu(self.ffn[1](self.ffn[0](X_next)))
                X_next = X_next + self.ffn[2](tmp) if len(self.ffn) > 2 else X_next
            else:
                X_next = X_next + self.ffn(X_next)
        return X_next

# --------------------------
# Positional encoding, encoder & forecasters 
# --------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        L = x.shape[1]
        return x + self.pe[:, :L, :]

class TSHMEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, n_layers=3, r=32, K=8, ff_hidden=512, use_pos=True):
        super().__init__()
        self.input_dim = input_dim
        self.embed = nn.Linear(input_dim, d_model)
        self.use_pos = use_pos
        if use_pos:
            self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([TSHMBlockSimple(d_model=d_model, r=r, K=K, ff_hidden=ff_hidden, causal=True) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.embed(x)
        if self.use_pos:
            h = self.pos(h)
        for l in self.layers:
            h = l(h)
        h = self.out_ln(h)
        return h


# -------------------------
# TSHMBlock
# -------------------------
class TSHMBlock(nn.Module):
    def __init__(
        self,
        d_model,
        r=32,
        K=8,
        ff_hidden=512,
        gate_kernel=3,
        per_channel_gate=False,
        gate_bias_init=-1.0,
        res_scale_init=0.2,
        causal=False,
    ):
        super().__init__()
        self.d = d_model
        self.causal = bool(causal)
        self.r = int(r)
        self.K = int(K)
        self.gate_kernel = int(gate_kernel)
        self.per_channel_gate = bool(per_channel_gate)

        self.U = nn.Linear(d_model, r, bias=False)
        self.V = nn.Linear(d_model, r, bias=False)
        self.A = nn.Linear(r, K, bias=True)
        self.B = nn.Linear(r, K, bias=True)
        self.c = nn.Parameter(torch.zeros(K))

        out_ch = d_model if per_channel_gate else 1
        self.gate_conv = nn.Conv1d(in_channels=d_model, out_channels=out_ch, kernel_size=self.gate_kernel, padding=0)
        nn.init.normal_(self.gate_conv.weight, std=0.02)
        nn.init.constant_(self.gate_conv.bias, gate_bias_init)

        self.res_scale = nn.Parameter(torch.tensor(float(res_scale_init)))
        self.layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4)

        self.pre_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ff_hidden),
            nn.GELU() if hasattr(nn, "GELU") else nn.Identity(),
            nn.Linear(ff_hidden, d_model),
        )

        self.register_buffer("_const_w", torch.tensor(0.1))
        self.register_buffer("_const_s", torch.tensor(0.1))

        for lin in (self.U, self.V, self.A, self.B):
            if hasattr(lin, "weight"):
                nn.init.xavier_normal_(lin.weight)

    def forward(self, X):
        B, L, d = X.shape
        assert d == self.d, f"Expected input dim {self.d}, got {d}"
        X_norm = self.pre_ln(X)
        phi = torch.tanh(X_norm)
        P = self.U(phi)
        S = torch.cumsum(P, dim=1)
        Q = self.V(phi)
        eA = self.A(S)
        eB = self.B(Q)
        e = eA + eB + self.c.view(1, 1, -1)
        e = torch.tanh(e)
        G = (self._const_w * (self._const_s ** 2)) * e
        M = torch.einsum("blk,kr->blr", G, self.A.weight)

        if self.causal:
            M_acc = torch.cumsum(M, dim=1)
            term1_pre = torch.einsum("blr,rd->bld", M_acc, self.U.weight)
            N_all = torch.einsum("blk,kr->blr", G, self.B.weight)
            N_acc = torch.cumsum(N_all, dim=1)
            term2_pre = torch.einsum("blr,rd->bld", N_acc, self.V.weight)
            grad = term1_pre + term2_pre
        else:
            M_rev = torch.flip(M, dims=[1])
            M_rev_cumsum = torch.cumsum(M_rev, dim=1)
            M_suf = torch.flip(M_rev_cumsum, dims=[1])
            term1_pre = torch.einsum("blr,rd->bld", M_suf, self.U.weight)
            N_all = torch.einsum("blk,kr->blr", G, self.B.weight)
            term2_pre = torch.einsum("blr,rd->bld", N_all, self.V.weight)
            grad = term1_pre + term2_pre

        g_in = X_norm.permute(0, 2, 1).contiguous()
        if self.causal:
            pad = self.gate_kernel - 1
            g_in_padded = F.pad(g_in, (pad, 0)) if pad > 0 else g_in
            g_raw = self.gate_conv(g_in_padded)
            g_raw = g_raw[..., -L:]
        else:
            pad_total = (self.gate_kernel - 1) // 2
            if pad_total > 0:
                g_in_padded = F.pad(g_in, (pad_total, pad_total))
                g_raw = self.gate_conv(g_in_padded)[..., :L]
            else:
                g_raw = self.gate_conv(g_in)[..., :L]

        g_raw = g_raw.permute(0, 2, 1).contiguous()
        if g_raw.shape[-1] == 1:
            g = torch.sigmoid(g_raw).expand(-1, -1, self.d)
        else:
            g = torch.sigmoid(g_raw)
        eps = 1e-6
        g = eps + (1.0 - eps) * g

        residual = (self.res_scale * g * grad) * self.layer_scale.view(1, 1, -1)
        X_next = X + residual
        X_next = X_next + self.ffn(X_next)
        return X_next

    def init_state(self, batch_size: int, device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device
        B = int(batch_size)
        pad = max(0, self.gate_kernel - 1)
        S = torch.zeros((B, self.r), dtype=torch.float32, device=device)
        M_pref = torch.zeros((B, self.r), dtype=torch.float32, device=device)
        N_pref = torch.zeros((B, self.r), dtype=torch.float32, device=device)
        if pad > 0:
            gate_buf = torch.zeros((B, self.d, pad), dtype=torch.float32, device=device)
        else:
            gate_buf = torch.zeros((B, self.d, 0), dtype=torch.float32, device=device)
        return {"S": S, "M_pref": M_pref, "N_pref": N_pref, "gate_buf": gate_buf}

    def forward_step(self, x_t, state):
        if not self.causal:
            raise RuntimeError("forward_step only supported in causal mode")
        B, d = x_t.shape
        assert d == self.d
        x_norm = self.pre_ln(x_t)
        phi_t = torch.tanh(x_norm)
        phi_prime = 1.0 - phi_t * phi_t
        P_t = self.U(phi_t)
        Q_t = self.V(phi_t)
        state["S"] = state["S"] + P_t
        e_t = self.A(state["S"]) + self.B(Q_t) + self.c.view(1, -1)
        e_t = torch.tanh(e_t)
        G_t = (self._const_w * (self._const_s ** 2)) * e_t
        M_t = torch.einsum("bk,kr->br", G_t, self.A.weight)
        N_t = torch.einsum("bk,kr->br", G_t, self.B.weight)
        state["M_pref"] = state["M_pref"] + M_t
        state["N_pref"] = state["N_pref"] + N_t
        term1 = torch.matmul(state["M_pref"], self.U.weight) * phi_prime
        term2 = torch.matmul(state["N_pref"], self.V.weight) * phi_prime
        grad_t = term1 + term2
        pad = max(0, self.gate_kernel - 1)
        if pad > 0:
            g_in_padded = torch.cat((state["gate_buf"], x_norm.unsqueeze(-1)), dim=-1)
            state["gate_buf"] = g_in_padded[..., 1:].detach()
        else:
            g_in_padded = x_norm.unsqueeze(-1)
        g_conv_out = self.gate_conv(g_in_padded)
        g_raw = g_conv_out.squeeze(-1)
        if g_raw.shape[-1] == 1:
            g = torch.sigmoid(g_raw).expand(-1, self.d)
            g = g.squeeze(-1) if g.dim() == 3 else g
        else:
            if g_raw.shape[-1] == self.d:
                g = torch.sigmoid(g_raw)
            else:
                g = torch.sigmoid(g_raw[:, :1]).expand(-1, self.d)
        eps = 1e-6
        g = eps + (1.0 - eps) * g
        residual = (self.res_scale * g * grad_t) * self.layer_scale.view(1, -1)
        x_next = x_t + residual
        x_next = x_next + self.ffn(x_next)
        return x_next, state


# -------------------------
# Inter-layer helper blocks and stacked wrapper
# -------------------------
class BetweenBlock(nn.Module):
    def __init__(self, d_model, dropout=0.0, activation="gelu", use_layernorm=True, causal=False):
        super().__init__()
        self.use_layernorm = use_layernorm
        self.causal = bool(causal)
        self.layernorm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()
        if activation == "gelu":
            self.act = nn.GELU() if hasattr(nn, "GELU") else nn.Identity()
        elif activation == "silu":
            self.act = nn.SiLU() if hasattr(nn, "SiLU") else nn.Identity()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()
        self.dropout = nn.Dropout(dropout) if (dropout and dropout > 0.0) else nn.Identity()

    def forward(self, x):
        if self.use_layernorm:
            x = self.layernorm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

    def forward_step(self, x_t):
        if self.use_layernorm:
            x_t = self.layernorm(x_t)
        x_t = self.act(x_t)
        x_t = self.dropout(x_t)
        return x_t


class GatedSkip(nn.Module):
    def __init__(self, d_model, init_gate=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_gate)))
        self.layer_scale = nn.Parameter(torch.ones(d_model) * 1e-4)

    def forward(self, x_prev, x_curr):
        g = torch.sigmoid(self.alpha)
        return x_curr * g + x_prev * (1.0 - g)

    def forward_step(self, x_prev_t, x_curr_t):
        g = torch.sigmoid(self.alpha)
        return x_curr_t * g + x_prev_t * (1.0 - g)


class TSHMStack(nn.Module):
    def __init__(self, num_layers, d_model, tshm_kwargs=None, between_kwargs=None, use_gated_skip=False, causal=False):
        super().__init__()
        tshm_kwargs = dict(tshm_kwargs or {})
        between_kwargs = dict(between_kwargs or {})
        tshm_kwargs.setdefault("causal", bool(causal))
        between_kwargs.setdefault("causal", bool(causal))
        self.num_layers = int(num_layers)
        self.d_model = int(d_model)
        self.causal = bool(causal)
        self.use_gated_skip = bool(use_gated_skip)
        self.layers = nn.ModuleList([TSHMBlock(d_model=d_model, **tshm_kwargs) for _ in range(self.num_layers)])
        between_defaults = dict(dropout=between_kwargs.get("dropout", 0.0), activation=between_kwargs.get("activation", "gelu"), use_layernorm=between_kwargs.get("use_layernorm", True), causal=between_kwargs.get("causal", False))
        self.betweens = nn.ModuleList([BetweenBlock(d_model=d_model, **between_defaults) for _ in range(max(0, self.num_layers - 1))])
        if self.use_gated_skip:
            self.gates = nn.ModuleList([GatedSkip(d_model=d_model, init_gate=0.0) for _ in range(max(0, self.num_layers - 1))])
        else:
            self.gates = None

    def forward(self, x):
        assert x.dim() == 3 and x.shape[-1] == self.d_model
        x_prev = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.betweens):
                x_between = self.betweens[i](x)
                if self.use_gated_skip:
                    x = self.gates[i](x_prev, x_between)
                else:
                    x = x_between
                x_prev = x
        return x

    def init_state(self, batch_size: int, device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device
        state_list = []
        for layer in self.layers:
            s = layer.init_state(batch_size=batch_size, device=device) if hasattr(layer, "init_state") else None
            state_list.append({"tshm": s, "between": None})
        return state_list

    def forward_step(self, x_t, state_list):
        assert self.causal, "forward_step requires causal=True"
        states = state_list
        x_prev_t = x_t
        for i, layer in enumerate(self.layers):
            s = states[i]
            if s is None or s.get("tshm", None) is None:
                raise RuntimeError("Missing per-layer TSHM streaming state; call init_state first")
            x_t, s_tshm = layer.forward_step(x_prev_t, s["tshm"])
            s["tshm"] = s_tshm
            if i < len(self.betweens):
                x_between_t = self.betweens[i].forward_step(x_t)
                if self.use_gated_skip:
                    x_t = self.gates[i].forward_step(x_prev_t, x_between_t)
                else:
                    x_t = x_between_t
                x_prev_t = x_t
        return x_t, states


# -------------------------
# Positional encoding and encoder stack
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        L = x.shape[1]
        return x + self.pe[:, :L, :]


class TSHMEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, n_layers=3, r=32, K=8, ff_hidden=512, use_pos=True, causal=False):
        super().__init__()
        self.input_dim = input_dim
        self.embed = nn.Linear(input_dim, d_model)
        self.use_pos = use_pos
        self.causal = causal
        if use_pos and not causal:
            self.pos = PositionalEncoding(d_model)
        else:
            self.pos = None
        tshm_kwargs = dict(r=r, K=K, ff_hidden=ff_hidden, gate_kernel=3, per_channel_gate=False, causal=causal)
        between_kwargs = dict(dropout=0.0, activation="relu", use_layernorm=False, causal=causal)
        self.stack = TSHMStack(num_layers=n_layers, d_model=d_model, tshm_kwargs=tshm_kwargs, between_kwargs=between_kwargs, use_gated_skip=True, causal=causal)
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.embed(x)
        if self.use_pos and self.pos is not None:
            h = self.pos(h)
        h = self.stack(h)
        h = self.out_ln(h)
        return h

    def init_stream_state(self, batch_size: int, device: torch.device = None):
        return self.stack.init_state(batch_size=batch_size, device=device)

    def forward_step(self, x_t, states):
        h = self.embed(x_t)
        h_out, new_states = self.stack.forward_step(h, states)
        h_out = self.out_ln(h_out)
        return h_out, new_states


class TSHMForecaster(nn.Module):
    def __init__(self, input_dim, out_dim, pred_len, d_model=256, n_layers=3, r=32, K=8, ff_hidden=512, use_pos=True):
        super().__init__()
        self.encoder = TSHMEncoder(input_dim=input_dim, d_model=d_model, n_layers=n_layers, r=r, K=K, ff_hidden=ff_hidden, use_pos=use_pos)
        #self.head = nn.Linear(d_model, pred_len * out_dim)
        self.out_dim = out_dim
        self.pred_len = pred_len

        dropout = 0.1

        self.head = nn.Sequential(
            #nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            #nn.Dropout(dropout),
            nn.Linear(256, pred_len * out_dim),
        )

    def forward(self, x):
        enc = self.encoder(x)
        #pooled = enc.mean(dim=1)
        pooled = enc.max(dim=1)[0]
        out = self.head(pooled)
        out = out.view(-1, self.pred_len, self.out_dim)
        return out
# --------------------------
# Utility helpers & runner (unchanged logic, with added dataset_class arg)
# --------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mse_mae(y_pred, y_true):
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    return mse, mae

def evaluate_forecast(model, loader, device, scaler_dict=None):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            out = out.cpu().numpy()
            y = y.cpu().numpy()
            if scaler_dict is not None:
                y_mean = scaler_dict['y_mean']
                y_std = scaler_dict['y_std']
                if y_mean.ndim == 2 and y_mean.shape[0] == 1:
                    y_mean = y_mean.reshape((1, 1, -1))
                    y_std = y_std.reshape((1, 1, -1))
                out = out * y_std + y_mean
                y = y * y_std + y_mean
            preds.append(out)
            trues.append(y)
    if len(preds) == 0:
        return (float('nan'), float('nan')), (np.zeros((0,0,0)), np.zeros((0,0,0)))
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return mse_mae(preds, trues), (preds, trues)

def predict_on_loader(model, loader, device, max_batches=None):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            out = model(x).cpu().numpy()
            preds.append(out)
            trues.append(y.cpu().numpy())
            if max_batches is not None and (i + 1) >= max_batches:
                break
    if len(preds) == 0:
        return np.zeros((0,0,0)), np.zeros((0,0,0))
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues

def diag_print_scaler_stats(train_ds, val_ds, name=""):
    print(f"--- {name} scaler / data stats ---")
    if hasattr(train_ds, 'y_mean'):
        print("train y_mean shape:", getattr(train_ds, 'y_mean').shape, "y_std shape:", getattr(train_ds, 'y_std').shape)
    if hasattr(val_ds, 'y_mean'):
        print("val y_mean shape:", getattr(val_ds, 'y_mean').shape, "y_std shape:", getattr(val_ds, 'y_std').shape)
    try:
        tr_vals = getattr(train_ds, 'data_y', None)
        val_vals = getattr(val_ds, 'data_y', None)
        if tr_vals is not None and val_vals is not None:
            print("train target: mean {:.4f} std {:.4f}".format(float(tr_vals.mean()), float(tr_vals.std())))
            print("val   target: mean {:.4f} std {:.4f}".format(float(val_vals.mean()), float(val_vals.std())))
    except Exception as e:
        print("couldn't compute raw train/val stats:", e)

def print_examples(model, loader, device, scaler_for_eval=None, n=5):
    model.eval()
    it = iter(loader)
    printed = 0
    with torch.no_grad():
        while printed < n:
            try:
                x, y = next(it)
            except StopIteration:
                break
            x = x.to(device)
            out = model(x).cpu().numpy()
            y_np = y.cpu().numpy()
            if scaler_for_eval is not None:
                y_mean = scaler_for_eval['y_mean']
                y_std = scaler_for_eval['y_std']
                if y_mean.ndim == 2 and y_mean.shape[0] == 1:
                    y_mean = y_mean.reshape((1, 1, -1))
                    y_std = y_std.reshape((1, 1, -1))
                out_orig = out * y_std + y_mean
                y_orig = y_np * y_std + y_mean
            else:
                out_orig, y_orig = out, y_np
            for b in range(min(out.shape[0], n-printed)):
                print(f"Example {printed + 1}:")
                print(" pred (first 6 steps):", np.round(out_orig[b].reshape(-1)[:6], 4))
                print(" true (first 6 steps):", np.round(y_orig[b].reshape(-1)[:6], 4))
                printed += 1
                if printed >= n:
                    break

def per_horizon_errors(preds, trues):
    if preds.size == 0:
        return np.array([])
    H = preds.shape[1]
    errors = []
    for h in range(H):
        diff = preds[:, h, :] - trues[:, h, :]
        mse = float(np.mean(np.square(diff)))
        mae = float(np.mean(np.abs(diff)))
        errors.append((mse, mae))
    return errors

def diagnose(model, train_loader, val_loader, device, scaler_for_eval, train_ds=None, val_ds=None, n_examples=5):
    print("=== DIAGNOSTIC REPORT ===")
    if train_ds is not None and val_ds is not None:
        diag_print_scaler_stats(train_ds, val_ds, name="ETT/generic")
    preds_tr_norm, trues_tr_norm = predict_on_loader(model, train_loader, device, max_batches=None)
    preds_val_norm, trues_val_norm = predict_on_loader(model, val_loader, device, max_batches=None)
    if preds_tr_norm.size > 0:
        tr_mse_norm, tr_mae_norm = mse_mae(preds_tr_norm, trues_tr_norm)
        print(f"Train normalized MSE={tr_mse_norm:.6f} MAE={tr_mae_norm:.6f}")
    if preds_val_norm.size > 0:
        val_mse_norm, val_mae_norm = mse_mae(preds_val_norm, trues_val_norm)
        print(f"Val   normalized MSE={val_mse_norm:.6f} MAE={val_mae_norm:.6f}")
    if scaler_for_eval is not None:
        def inv_transform(arr):
            y_mean = scaler_for_eval['y_mean']
            y_std = scaler_for_eval['y_std']
            if y_mean.ndim == 2 and y_mean.shape[0] == 1:
                y_mean = y_mean.reshape((1,1,-1))
                y_std = y_std.reshape((1,1,-1))
            return arr * y_std + y_mean
        preds_tr_orig = inv_transform(preds_tr_norm) if preds_tr_norm.size>0 else preds_tr_norm
        trues_tr_orig = inv_transform(trues_tr_norm) if trues_tr_norm.size>0 else trues_tr_norm
        preds_val_orig = inv_transform(preds_val_norm) if preds_val_norm.size>0 else preds_val_norm
        trues_val_orig = inv_transform(trues_val_norm) if trues_val_norm.size>0 else trues_val_norm
        if preds_tr_orig.size > 0:
            tr_mse_orig, tr_mae_orig = mse_mae(preds_tr_orig, trues_tr_orig)
            print(f"Train original-scale MSE={tr_mse_orig:.6f} MAE={tr_mae_orig:.6f}")
        if preds_val_orig.size > 0:
            val_mse_orig, val_mae_orig = mse_mae(preds_val_orig, trues_val_orig)
            print(f"Val   original-scale MSE={val_mse_orig:.6f} MAE={val_mae_orig:.6f}")
        errs = per_horizon_errors(preds_val_orig, trues_val_orig)
        if len(errs) > 0:
            print("Per-horizon errors (val, original scale):")
            for h, (mse_h, mae_h) in enumerate(errs[:min(20, len(errs))]):
                print(f" h={h+1:3d}  MSE={mse_h:.6f}  MAE={mae_h:.6f}")
    print("=== some example predictions (original scale) ===")
    print_examples(model, val_loader, device, scaler_for_eval=scaler_for_eval, n=n_examples)
    print("=== end diagnostics ===")

# --------------------------
# Main CLI + loop (keeps your original logic; added --dataset_class arg)
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing dataset or path to single CSV")
    parser.add_argument("--dataset", type=str, default="ETTh1", help="Dataset name (ETTh1, ETTh2, ETTm1, ETTm2, custom)")
    parser.add_argument("--dataset_class", type=str, default=None,
                        help="explicit dataset class to use (case-insensitive): ETT_hour, ETT_minute, M4, Custom, ForecastCSV. Overrides auto-detect.")
    parser.add_argument("--input_len", type=int, default=2048, help="History length (if not provided uses recommended per-dataset)")
    parser.add_argument("--pred_len", type=int, default=168, help="Forecast horizon")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=784)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--r", type=int, default=124)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--ff_hidden", type=int, default=256)
    parser.add_argument("--model", choices=["tshm"], default="tshm")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval_horizons", nargs="+", type=int, default=[24,48,96, 288,168,336,672,720,960])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="ETT csv filename when using ETT dataset (if --data_dir is a folder)")
    parser.add_argument("--features", type=str, default="S", help="Features M/S/MS for ETT")
    parser.add_argument("--target", type=str, default="OT", help="Target column name for ETT when features='S'")
    args = parser.parse_args()

    set_seed(args.seed)

    recommended_input = {
        "ETTh1": 96, "ETTh2": 96, "ETTm1": 96, "ETTm2": 96,
    }
    input_len = args.input_len if args.input_len is not None else recommended_input.get(args.dataset, 96)
    pred_len = args.pred_len

    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")

    # If user provided a filepath to a CSV, canonicalize: use parent as data_dir and remember csv filename
    single_csv_provided = False
    provided_csv_path = None
    if data_path.is_file():
        single_csv_provided = True
        provided_csv_path = data_path
        data_dir = data_path.parent
    else:
        data_dir = data_path

    # AUTO-DETECT dataset from provided single CSV filename unless dataset_class overrides
    if single_csv_provided:
        try:
            bn = provided_csv_path.name.lower()
            if args.dataset_class is None:
                if (args.dataset is None) or (not args.dataset.startswith("ETT")) or args.dataset == "custom":
                    m = re.search(r'(etth|ettm)(\d+)', bn)
                    if m:
                        prefix = m.group(1)
                        num = m.group(2)
                        if prefix == 'etth':
                            args.dataset = f"ETTh{num}"
                        else:
                            args.dataset = f"ETTm{num}"
                    else:
                        if 'm4' in bn:
                            args.dataset = "M4"
                        elif 'weather' in bn:
                            args.dataset = "WEATHER"
                        elif 'ett' in bn:
                            m2 = re.search(r'ett[_\-]?(\d+)', bn)
                            if m2:
                                args.dataset = f"ETT{m2.group(1)}"
                            else:
                                args.dataset = "ETT"
                    if args.dataset != "custom":
                        print(f"[loader] auto-set args.dataset to '{args.dataset}' based on provided CSV name '{provided_csv_path.name}'")
        except Exception:
            pass

    # Normalize dataset_class argument
    dataset_class_arg = args.dataset_class.lower() if (args.dataset_class is not None) else None

    train_ds_raw = None
    val_ds_raw = None

    # If user explicitly requested a dataset_class, honor it (M4, ETT_hour, ETT_minute, Custom, ForecastCSV)
    if dataset_class_arg in ("ett_hour", "ett_hour".lower(), "etth", "ett_hour"):
        chosen_kind = "ETT_hour"
    elif dataset_class_arg in ("ett_minute", "ett_minute".lower(), "ettm", "ett_minute"):
        chosen_kind = "ETT_minute"
    elif dataset_class_arg in ("m4", "dataset_m4"):
        chosen_kind = "M4"
    elif dataset_class_arg in ("custom", "dataset_custom"):
        chosen_kind = "Custom"
    elif dataset_class_arg in ("forecastcsv", "forecast_csv"):
        chosen_kind = "ForecastCSV"
    else:
        chosen_kind = None  # not explicitly provided

    # Branch: ETT dataset uses THUML borders / scaler behavior OR explicit dataset_class
    if (args.dataset.startswith("ETT")) or (chosen_kind in ("ETT_hour", "ETT_minute")) or (chosen_kind in ("M4", "Custom", "ForecastCSV")):
        if not hasattr(args, "augmentation_ratio"):
            args.augmentation_ratio = 0

        # determine CSV to load: prefer provided single CSV, else use args.data_path inside data_dir
        if single_csv_provided:
            csv_name = provided_csv_path.name
            root_path = str(provided_csv_path.parent)
        else:
            csv_name = args.data_path
            root_path = str(data_dir)
        print(f"[loader] Using ETT/M4/Custom loader with file {csv_name} in {root_path} (THUML style split)")

        # Decide which loader to use:
        use_minute = False
        use_hour = False
        use_m4 = False
        use_custom = False
        use_forecastcsv = False

        if chosen_kind == "ETT_minute":
            use_minute = True
        elif chosen_kind == "ETT_hour":
            use_hour = True
        elif chosen_kind == "M4":
            use_m4 = True
        elif chosen_kind == "Custom":
            use_custom = True
        elif chosen_kind == "ForecastCSV":
            use_forecastcsv = True
        else:
            # fallback: decide from args.dataset
            if args.dataset.startswith("ETTm"):
                use_minute = True
            elif args.dataset.startswith("ETTh"):
                use_hour = True
            elif args.dataset.upper().startswith("M4"):
                use_m4 = True
            elif args.dataset.lower().startswith("custom"):
                use_custom = True
            else:
                # default ETT hour if ambiguous
                use_hour = True

        if use_m4:
            # Build M4 train/val/test datasets (Dataset_M4 expects a size tuple)
            # We'll instantiate train/val/test using Dataset_M4 with flags 'train','val','test' if possible.
            # Note: M4 loader internals depend on data_provider.m4; keep minimal wrapper behavior.
            train_ds_raw = Dataset_M4(args=args, root_path=root_path, flag='train', size=(input_len, 0, pred_len))
            val_ds_raw = Dataset_M4(args=args, root_path=root_path, flag='train', size=(input_len, 0, pred_len))
            test_ds_raw = Dataset_M4(args=args, root_path=root_path, flag='train', size=(input_len, 0, pred_len))
        elif use_custom:
            train_ds_raw = Dataset_Custom(args=args, root_path=root_path, flag='train', size=(input_len, 0, pred_len),
                                          features=args.features, data_path=csv_name, target=args.target, scale=True)
            val_ds_raw = Dataset_Custom(args=args, root_path=root_path, flag='val', size=(input_len, 0, pred_len),
                                        features=args.features, data_path=csv_name, target=args.target, scale=True)
            test_ds_raw = Dataset_Custom(args=args, root_path=root_path, flag='test', size=(input_len, 0, pred_len),
                                         features=args.features, data_path=csv_name, target=args.target, scale=True)
        elif use_forecastcsv:
            # Use ForecastCSVSequence (single CSV split) to produce train/val/test partitions
            # We will split the single CSV into 80/10/10 contiguous partitions
            provided_csv_fp = os.path.join(root_path, csv_name)
            if not os.path.exists(provided_csv_fp):
                raise FileNotFoundError(f"CSV file for ForecastCSV not found: {provided_csv_fp}")
            full_df = pd.read_csv(provided_csv_fp)
            x_cols = [c for c in full_df.columns if c.startswith("x_")]
            y_cols = [c for c in full_df.columns if c.startswith("y_")]
            if not x_cols:
                possible_y = [c for c in full_df.columns if ("target" in c.lower()) or (c.startswith("y_")) or (c.lower() == "y") or (c.lower() == "value")]
                if possible_y:
                    y_cols = possible_y
                    x_cols = [c for c in full_df.columns if c not in y_cols]
                else:
                    y_cols = [full_df.columns[-1]]
                    x_cols = list(full_df.columns[:-1])
            N = len(full_df)
            n_train = int(0.8 * N)
            n_val = int(0.1 * N)
            def make_ds_from_slices(xdf, ydf):
                ds = ForecastCSVSequence.__new__(ForecastCSVSequence)
                ds.data_dir = Path(root_path)
                ds.input_len = input_len
                ds.pred_len = pred_len
                ds.stride = 1
                ds.normalize = True
                ds.X_df = xdf.reset_index(drop=True)
                ds.Y_df = ydf.reset_index(drop=True)
                ds.X = ds.X_df.values.astype(np.float32)
                ds.Y = ds.Y_df.values.astype(np.float32)
                ds.x_mean = ds.X.mean(axis=0, keepdims=True)
                ds.x_std = ds.X.std(axis=0, keepdims=True) + 1e-6
                ds.y_mean = ds.Y.mean(axis=0, keepdims=True)
                ds.y_std = ds.Y.std(axis=0, keepdims=True) + 1e-6
                T = ds.X.shape[0]
                max_start = T - (ds.input_len + ds.pred_len)
                ds.starts = list(range(0, max_start + 1, ds.stride)) if max_start >= 0 else []
                return ds
            X_df = full_df[x_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
            Y_df = full_df[y_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
            train_ds = make_ds_from_slices(X_df.iloc[:n_train], Y_df.iloc[:n_train])
            val_ds = make_ds_from_slices(X_df.iloc[n_train:n_train+n_val], Y_df.iloc[n_train:n_train+n_val])
            test_ds = make_ds_from_slices(X_df.iloc[n_train+n_val:], Y_df.iloc[n_train+n_val:])
            # set up loaders directly and skip ETT wrapping below
            scaler_for_eval = {'y_mean': train_ds.y_mean, 'y_std': train_ds.y_std}
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
            train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=2)
            sample_inp, sample_tgt = next(iter(train_loader))
            input_dim = sample_inp.shape[-1]
            out_dim = sample_tgt.shape[-1]
            print(f"[loader] ForecastCSV split -> train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")
            # continue to model build below
            train_ds_raw = None
            val_ds_raw = None
        else:
            # Default ETT hour/minute
            if use_minute:
                train_ds_raw = Dataset_ETT_minute(args=args, root_path=root_path, flag='train', size=(input_len, 0, pred_len),
                                                 features=args.features, data_path=csv_name, target=args.target, scale=True)
                val_ds_raw = Dataset_ETT_minute(args=args, root_path=root_path, flag='val', size=(input_len, 0, pred_len),
                                               features=args.features, data_path=csv_name, target=args.target, scale=True)
                test_ds_raw = Dataset_ETT_minute(args=args, root_path=root_path, flag='test', size=(input_len, 0, pred_len),
                                                features=args.features, data_path=csv_name, target=args.target, scale=True)
            else:
                train_ds_raw = Dataset_ETT_hour(args=args, root_path=root_path, flag='train', size=(input_len, 0, pred_len),
                                               features=args.features, data_path=csv_name, target=args.target, scale=True)
                val_ds_raw = Dataset_ETT_hour(args=args, root_path=root_path, flag='val', size=(input_len, 0, pred_len),
                                             features=args.features, data_path=csv_name, target=args.target, scale=True)
                test_ds_raw = Dataset_ETT_hour(args=args, root_path=root_path, flag='test', size=(input_len, 0, pred_len),
                                              features=args.features, data_path=csv_name, target=args.target, scale=True)

                print("len(train_ds_raw), len(val_ds_raw), len(test_ds_raw)", len(train_ds_raw), len(val_ds_raw), len(test_ds_raw))

        # If forecastcsv handled above, we already created loaders and scalers and set input_dim/out_dim
        if not (dataset_class_arg and dataset_class_arg in ("forecastcsv", "forecast_csv")):
            # Print the target variable to demonstrate target name (as requested)
            print(f"[ETT/M4 loader] target variable name (args.target): {getattr(train_ds_raw, 'target', args.target)}")

            # --- DIAGNOSTIC PROOFS ABOUT TARGET / FEATURES ---
            try:
                csv_fp = os.path.join(root_path, csv_name)
                df_full = pd.read_csv(csv_fp)
                print("[ETT loader] CSV columns (first/last):", list(df_full.columns[:5]), "...", list(df_full.columns[-5:]))
                csv_last = df_full.columns[-1]
                print(f"[ETT loader] CSV last-column name: '{csv_last}'")
                print(f"[ETT loader] args.target='{args.target}'  => matches CSV last column? {csv_last == args.target}")

                print("[ETT loader] train_ds_raw.data_x shape:", getattr(train_ds_raw, "data_x", None).shape)
                print("[ETT loader] train_ds_raw.data_y shape:", getattr(train_ds_raw, "data_y", None).shape)

                train_scaler = train_ds_raw.get_scaler()
                try:
                    print("[ETT loader] train scaler mean shape:", train_scaler.mean_.shape, "scale shape:", train_scaler.scale_.shape)
                    print("[ETT loader] scaler_for_eval (first few): mean:", np.round(train_scaler.mean_.reshape(-1)[:5],6), " scale:", np.round(train_scaler.scale_.reshape(-1)[:5],6))
                except Exception:
                    try:
                        print("[ETT loader] train scaler type/info:", type(train_scaler))
                    except Exception:
                        pass

                if isinstance(train_ds_raw, Dataset_ETT_hour):
                    border1s = [0, 12 * 30 * 24 - train_ds_raw.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - train_ds_raw.seq_len]
                    border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
                else:
                    border1s = [0, 12 * 30 * 24 * 4 - train_ds_raw.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - train_ds_raw.seq_len]
                    border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
                b1 = border1s[train_ds_raw.set_type]
                b2 = border2s[train_ds_raw.set_type]
                print(f"[ETT loader] expected partition slice rows: [{b1}:{b2}] (len={b2-b1})")

                raw_last_col_vals = df_full[csv_last].iloc[b1:b2].to_numpy(dtype=np.float32)
                print(f"[ETT loader] CSV last-col slice shape: {raw_last_col_vals.shape}  (first 5):", raw_last_col_vals[:5])

                try:
                    inv_all = train_scaler.inverse_transform(train_ds_raw.data_y)
                    inv_last = inv_all[:, -1]
                    print(f"[ETT loader] inv(data_y) last-col shape: {inv_last.shape}  (first 5):", inv_last[:5])
                    if inv_last.shape[0] == raw_last_col_vals.shape[0]:
                        diff = inv_last - raw_last_col_vals
                        print("[ETT loader] mean(abs(inv_last - raw_csv_last)):", float(np.mean(np.abs(diff))))
                        print("[ETT loader] max(abs(inv_last - raw_csv_last)) :", float(np.max(np.abs(diff))))
                    else:
                        print("[ETT loader] WARNING: shape mismatch between inv_last and raw CSV slice:", inv_last.shape, raw_last_col_vals.shape)
                except Exception as e:
                    print("[ETT loader] inverse-transform diagnostic skipped/failed:", e)
            except Exception as e:
                print("[ETT loader] Diagnostic check failed:", e)

            # Build scaler_for_eval
            train_scaler = train_ds_raw.get_scaler()
            if hasattr(train_scaler, "mean_") and hasattr(train_scaler, "scale_"):
                scaler_dict = {}
                scaler_dict['y_mean'] = train_scaler.mean_.reshape((1, -1)).astype(np.float32)
                scaler_dict['y_std'] = (train_scaler.scale_.reshape((1, -1)).astype(np.float32) + 1e-6)
            else:
                try:
                    scaler_dict = {'y_mean': np.asarray(train_scaler).reshape((1, -1)).astype(np.float32), 'y_std': np.ones((1, train_ds_raw.data_y.shape[1]), dtype=np.float32)}
                except Exception:
                    scaler_dict = {'y_mean': np.zeros((1, train_ds_raw.data_y.shape[1]), dtype=np.float32), 'y_std': np.ones((1, train_ds_raw.data_y.shape[1]), dtype=np.float32)}

            # Wrap datasets for dataloader (compatible with forecaster expectation)
            class _WrappedETTDataset(Dataset):
                def __init__(self, ett_ds):
                    self.ett_ds = ett_ds
                def __len__(self):
                    return len(self.ett_ds)
                def __getitem__(self, idx):
                    items = self.ett_ds[idx]
                    if isinstance(items, (tuple, list)):
                        if len(items) == 2:
                            x, y = items
                        else:
                            x, y = items[0], items[1]
                    else:
                        raise RuntimeError("Unexpected item type from underlying ETT dataset.")
                    x = x.astype(np.float32)
                    return torch.tensor(x, dtype=torch.float32), torch.tensor(y[-pred_len:].astype(np.float32), dtype=torch.float32)

            train_ds = _WrappedETTDataset(train_ds_raw)
            val_ds = _WrappedETTDataset(val_ds_raw)
            test_ds = _WrappedETTDataset(test_ds_raw)

            scaler_for_eval = {'y_mean': scaler_dict['y_mean'], 'y_std': scaler_dict['y_std']}

            sample_x, sample_y = train_ds[0]
            input_dim = sample_x.shape[-1]
            out_dim = sample_y.shape[-1]
            print(f"[ETT loader] train samples: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
            train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=2)

    else:
        # Fallback to original generic CSV / folder behavior (kept largely unchanged)
        if single_csv_provided:
            print(f"[loader] Single CSV provided at {provided_csv_path}. Splitting into train/val/test contiguous 80/10/10.")
            df = pd.read_csv(provided_csv_path)
            x_cols = [c for c in df.columns if c.startswith("x_")]
            y_cols = [c for c in df.columns if c.startswith("y_")]
            if not x_cols:
                possible_y = [c for c in df.columns if ("target" in c.lower()) or (c.startswith("y_")) or (c.lower() == "y") or (c.lower() == "value")]
                if possible_y:
                    y_cols = possible_y
                    x_cols = [c for c in df.columns if c not in y_cols]
                else:
                    y_cols = [df.columns[-1]]
                    x_cols = list(df.columns[:-1])
            X_df = df[x_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
            Y_df = df[y_cols].copy().apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
            N = X_df.shape[0]
            n_train = int(0.8 * N)
            n_val = int(0.1 * N)
            def make_ds_from_slices(xdf, ydf):
                ds = ForecastCSVSequence.__new__(ForecastCSVSequence)
                ds.data_dir = provided_csv_path.parent
                ds.input_len = input_len
                ds.pred_len = pred_len
                ds.stride = 1
                ds.normalize = True
                ds.X_df = xdf.reset_index(drop=True)
                ds.Y_df = ydf.reset_index(drop=True)
                ds.X = ds.X_df.values.astype(np.float32)
                ds.Y = ds.Y_df.values.astype(np.float32)
                ds.x_mean = ds.X.mean(axis=0, keepdims=True)
                ds.x_std = ds.X.std(axis=0, keepdims=True) + 1e-6
                ds.y_mean = ds.Y.mean(axis=0, keepdims=True)
                ds.y_std = ds.Y.std(axis=0, keepdims=True) + 1e-6
                T = ds.X.shape[0]
                max_start = T - (ds.input_len + ds.pred_len)
                ds.starts = list(range(0, max_start + 1, ds.stride)) if max_start >= 0 else []
                return ds
            train_ds = make_ds_from_slices(X_df.iloc[:n_train], Y_df.iloc[:n_train])
            val_ds = make_ds_from_slices(X_df.iloc[n_train:n_train+n_val], Y_df.iloc[n_train:n_train+n_val])
            test_ds = make_ds_from_slices(X_df.iloc[n_train+n_val:], Y_df.iloc[n_train+n_val:])
            scaler = train_ds.get_scaler_dict()
            for ds in (val_ds, test_ds):
                if hasattr(ds, 'x_mean'):
                    ds.x_mean = scaler['x_mean']
                    ds.x_std = scaler['x_std']
                    ds.y_mean = scaler['y_mean']
                    ds.y_std = scaler['y_std']
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
            train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=2)
            sample_inp, sample_tgt = next(iter(train_loader))
            input_dim = sample_inp.shape[-1]
            out_dim = sample_tgt.shape[-1]
            scaler_for_eval = {'y_mean': train_ds.y_mean, 'y_std': train_ds.y_std}
            print(f"[loader] Single CSV split -> train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")
        else:
            # If partitioned folder exists, reuse original logic from your script
            def load_partition_generic(part_name):
                part_dir = data_dir / part_name
                if part_dir.exists():
                    return ForecastCSVSequence(part_dir, input_len=input_len, pred_len=pred_len, stride=1, normalize=True)
                else:
                    sx = data_dir / f"{part_name}_df_x.csv"
                    sy = data_dir / f"{part_name}_df_y.csv"
                    if sx.exists() and sy.exists():
                        ds = ForecastCSVSequence.__new__(ForecastCSVSequence)
                        ds.data_dir = data_dir
                        ds.input_len = input_len
                        ds.pred_len = pred_len
                        ds.stride = 1
                        ds.normalize = True
                        ds.X_df = pd.read_csv(sx).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
                        ds.Y_df = pd.read_csv(sy).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
                        ds.X = ds.X_df.values.astype(np.float32)
                        ds.Y = ds.Y_df.values.astype(np.float32)
                        ds.x_mean = ds.X.mean(axis=0, keepdims=True)
                        ds.x_std = ds.X.std(axis=0, keepdims=True) + 1e-6
                        ds.y_mean = ds.Y.mean(axis=0, keepdims=True)
                        ds.y_std = ds.Y.std(axis=0, keepdims=True) + 1e-6
                        T = ds.X.shape[0]
                        max_start = T - (ds.input_len + ds.pred_len)
                        ds.starts = list(range(0, max_start + 1, ds.stride)) if max_start >= 0 else []
                        return ds
                    x_all = data_dir / "df_x.csv"
                    y_all = data_dir / "df_y.csv"
                    if x_all.exists() and y_all.exists():
                        df_x = pd.read_csv(x_all).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
                        df_y = pd.read_csv(y_all).apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float32)
                        N = df_x.shape[0]
                        n_train = int(0.8 * N)
                        n_val = int(0.1 * N)
                        def make_ds_from_slices(xdf, ydf):
                            ds = ForecastCSVSequence.__new__(ForecastCSVSequence)
                            ds.data_dir = data_dir
                            ds.input_len = input_len
                            ds.pred_len = pred_len
                            ds.stride = 1
                            ds.normalize = True
                            ds.X_df = xdf
                            ds.Y_df = ydf
                            ds.X = ds.X_df.values.astype(np.float32)
                            ds.Y = ds.Y_df.values.astype(np.float32)
                            ds.x_mean = ds.X.mean(axis=0, keepdims=True)
                            ds.x_std = ds.X.std(axis=0, keepdims=True) + 1e-6
                            ds.y_mean = ds.Y.mean(axis=0, keepdims=True)
                            ds.y_std = ds.Y.std(axis=0, keepdims=True) + 1e-6
                            T = ds.X.shape[0]
                            max_start = T - (ds.input_len + ds.pred_len)
                            ds.starts = list(range(0, max_start + 1, ds.stride)) if max_start >= 0 else []
                            return ds
                        train_ds = make_ds_from_slices(df_x.iloc[:n_train].reset_index(drop=True), df_y.iloc[:n_train].reset_index(drop=True))
                        val_ds = make_ds_from_slices(df_x.iloc[n_train:n_train+n_val].reset_index(drop=True), df_y.iloc[n_train:n_train+n_val].reset_index(drop=True))
                        test_ds = make_ds_from_slices(df_x.iloc[n_train+n_val:].reset_index(drop=True), df_y.iloc[n_train+n_val:].reset_index(drop=True))
                        return {"train": train_ds, "validation": val_ds, "test": test_ds}
                    raise FileNotFoundError(f"Could not find partition {part_name} under {data_dir}")

            if (data_dir / "train").exists():
                train_ds = load_partition_generic("train")
                val_ds = load_partition_generic("validation")
                test_ds = load_partition_generic("test")
            else:
                tmp = load_partition_generic("train")
                if isinstance(tmp, dict):
                    train_ds, val_ds, test_ds = tmp['train'], tmp['validation'], tmp['test']
                else:
                    raise FileNotFoundError(f"Could not find partition train under {data_dir}. Provide train/validation/test or a df_x.csv+df_y.csv or a single CSV file.")

            scaler = train_ds.get_scaler_dict()
            for ds in (val_ds, test_ds):
                if hasattr(ds, 'x_mean'):
                    ds.x_mean = scaler['x_mean']
                    ds.x_std = scaler['x_std']
                    ds.y_mean = scaler['y_mean']
                    ds.y_std = scaler['y_std']
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
            train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=max(1, args.batch_size//2), shuffle=False, num_workers=2)
            sample_inp, sample_tgt = next(iter(train_loader))
            input_dim = sample_inp.shape[-1]
            out_dim = sample_tgt.shape[-1]
            scaler_for_eval = {'y_mean': train_ds.y_mean, 'y_std': train_ds.y_std}
            print(f"[loader] train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}")

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)
    print(f"Dataset: {args.dataset} | input_len={input_len} pred_len={pred_len}")

    # build model
    if args.model == "tshm":
        model = TSHMForecaster(input_dim=input_dim, out_dim=out_dim, pred_len=pred_len,
                               d_model=args.d_model, n_layers=args.n_layers, r=args.r, K=args.K, ff_hidden=args.ff_hidden, use_pos=True)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        train_loss = total_loss / max(1, n_batches)

        (train_mse_norm, train_mae_norm), _ = evaluate_forecast(model, train_eval_loader, device, scaler_dict=None)
        (val_mse_norm, val_mae_norm), _ = evaluate_forecast(model, val_loader, device, scaler_dict=None)
        (train_mse_orig, train_mae_orig), _ = evaluate_forecast(model, train_eval_loader, device, scaler_dict=scaler_for_eval)
        (val_mse_orig, val_mae_orig), _ = evaluate_forecast(model, val_loader, device, scaler_dict=scaler_for_eval)

        dt = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss(norm)={train_loss:.6f} "
            f"| train_mse(norm)={train_mse_norm:.6f} train_mae(norm)={train_mae_norm:.6f} "
            f"| val_mse(norm)={val_mse_norm:.6f} val_mae(norm)={val_mae_norm:.6f} "
            f"| val_mse(orig)={val_mse_orig:.6f} val_mae(orig)={val_mae_orig:.6f} | {dt:.1f}s"
        )

        if val_mse_orig < best_val:
            best_val = val_mse_orig
            best_state = model.state_dict()
            torch.save(best_state, f"best_{args.model}_{args.dataset}.pth")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Diagnostics
    try:
        diagnose(model, train_loader, val_loader, device, scaler_for_eval, train_ds=train_ds_raw, val_ds=val_ds_raw, n_examples=8)
    except Exception:
        try:
            diagnose(model, train_loader, val_loader, device, scaler_for_eval, train_ds=None, val_ds=None, n_examples=8)
        except Exception:
            print("[diagnose] diagnostics failed.")

    # Final test evaluation
    results = {}
    for horizon in args.eval_horizons:
        if horizon != pred_len:
            print(f"[warn] Different horizon {horizon} requested; this runner only evaluates at pred_len={pred_len}. Skipping horizon {horizon}.")
            continue

        preds_norm, trues_norm = predict_on_loader(model, test_loader, device)
        if preds_norm.size == 0:
            print("[warn] No test predictions (empty test set).")
            results[horizon] = {"mse": float('nan'), "mae": float('nan')}
            continue
        mse_norm, mae_norm = mse_mae(preds_norm, trues_norm)

        y_mean = scaler_for_eval['y_mean']
        y_std = scaler_for_eval['y_std']
        if y_mean.ndim == 2 and y_mean.shape[0] == 1:
            y_mean = y_mean.reshape((1,1,-1))
            y_std = y_std.reshape((1,1,-1))
        preds_orig = preds_norm * y_std + y_mean
        trues_orig = trues_norm * y_std + y_mean
        mse_orig, mae_orig = mse_mae(preds_orig, trues_orig)

        results[horizon] = {"mse": float(mse_orig), "mae": float(mae_orig)}
        out_csv = f"predictions_{args.model}_{args.dataset}_h{horizon}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["sample_idx", "step"]
            for d in range(trues_orig.shape[2]):
                header.append(f"true_dim{d}")
            for d in range(preds_orig.shape[2]):
                header.append(f"pred_dim{d}")
            w.writerow(header)
            n_save = min(200, preds_orig.shape[0])
            for i in range(n_save):
                for s in range(preds_orig.shape[1]):
                    row = [i, s]
                    row += [float(x) for x in trues_orig[i, s].tolist()]
                    row += [float(x) for x in preds_orig[i, s].tolist()]
                    w.writerow(row)
        print(f"Horizon {horizon}: MSE(orig)={mse_orig:.6f}, MAE(orig)={mae_orig:.6f} | saved {out_csv}")
        print(f"Horizon {horizon}: MSE(norm)={mse_norm:.6f}, MAE(norm)={mae_norm:.6f}")

    print("All results:", results)

if __name__ == "__main__":
    main()


#python3 tshm_transformer70.py --data_dir /workspace/EETok/weather/weather.csv --model tshm --epochs 7 --dataset_class ETT_minute

#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTm1.csv --model tshm --epochs 7 --dataset_class ETT_minute

#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTh2.csv --model tshm --epochs 15 --dataset_class ETT_hour

#python3 tshm_transformer70.py --data_dir /workspace/EETok/weather/weather.csv --model tshm --epochs 8 --dataset_class custom --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296  --input_len 2096 --pred_len 720

#python3 tshm_transformer70.py --data_dir /workspace/EETok/weather/weather.csv --model tshm --epochs 8 --dataset_class custom --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296  --input_len 192 --pred_len 48


#python3 tshm_transformer70.py --data_dir /workspace/time_serie --model tshm --epochs 2 --dataset_class custom --batch_size 8 --d_model 128 --ff_hidden 128 --n_layers 4 --r 64 --input_len 86 --pred_len 48 

#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTm1.csv --model tshm --epochs 8 --dataset_class ETT_minute --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --K 8 --input_len 288 --pred_len 48 --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTm1.csv --model tshm --epochs 30 --dataset_class ETT_minute --batch_size 32 --d_model 256 --ff_hidden 32 --n_layers 2 --r 296 --K 8 --input_len 48 --pred_len 24 --lr 1e-5


#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTm1.csv --model tshm --epochs 30 --dataset_class ETT_minute --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --K 8 --input_len 1728 --pred_len 288 --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTm1.csv --model tshm --epochs 25 --dataset_class ETT_minute --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --K 8 --input_len 1024 --pred_len 672 --lr 1e-5


#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTh2.csv --model tshm --epochs 45 --dataset_class ETT_hour --batch_size 16 --d_model 256 --ff_hidden 32 --n_layers 2 --r 296 --input_len 288 --pred_len 48 --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTh2.csv --model tshm --epochs 8 --dataset_class ETT_hour --batch_size 16 --d_model 256 --ff_hidden 32 --n_layers 2 --r 296 --input_len 672 --pred_len 168 --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTh2.csv --model tshm --epochs 30 --dataset_class ETT_hour --batch_size 16 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 2016 --pred_len 336 --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTh2.csv --model tshm --epochs 30 --dataset_class ETT_hour --batch_size 16 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 48 --pred_len 24 --lr 1e-5








#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTh1.csv --model tshm --epochs 45 --dataset_class ETT_hour --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 24 --pred_len 24 --lr 8e-6

#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTh1.csv --model tshm --epochs 30 --dataset_class ETT_hour --batch_size 16 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 48 --pred_len 24 --lr 1e-5


#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTh1.csv --model tshm --epochs 26 --dataset_class ETT_hour --batch_size 16 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 336 --pred_len 168 --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTh1.csv --model tshm --epochs 45 --dataset_class ETT_hour --batch_size 16 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 672 --pred_len 336 --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/ETTh1.csv --model tshm --epochs 45 --dataset_class ETT_hour --batch_size 16 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 1024 --pred_len 720 --lr 1e-5





#python3 tshm_transformer70.py --data_dir /workspace/EETok/weather/weather.csv --model tshm --epochs 8 --dataset_class custom --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 96 --pred_len 48 --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/weather/weather.csv --model tshm --epochs 8 --dataset_class custom --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 48 --pred_len 24 --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/weather/weather.csv --model tshm --epochs 8 --dataset_class custom --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 336 --pred_len 168 --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/weather/weather.csv --model tshm --epochs 8 --dataset_class custom --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 672 --pred_len 336 --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/weather/weather.csv --model tshm --epochs 8 --dataset_class custom --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 2 --r 296 --input_len 1024 --pred_len 720 --lr 1e-5





#python3 tshm_transformer70.py --data_dir /workspace/EETok/electricity/electricity.csv --model tshm --epochs 15 --dataset_class custom --batch_size 8 --d_model 128 --ff_hidden 128 --n_layers 4 --r 64 --input_len 86 --pred_len 48  --lr 1e-5

#python3 tshm_transformer70.py --data_dir /workspace/EETok/electricity/electricity.csv --model tshm --epochs 13 --dataset_class custom --batch_size 16 --d_model 128 --ff_hidden 32 --n_layers 2 --r 128 --input_len 192 --pred_len 48  --lr 6e-4

#python3 tshm_transformer70.py --data_dir /workspace/EETok/electricity/electricity.csv --model tshm --epochs 13 --dataset_class custom --batch_size 32 --d_model 128 --ff_hidden 32 --n_layers 2 --r 128 --input_len 1024 --pred_len 720  --lr 6e-4


#python3 tshm_transformer86.py --data_dir /workspace/EETok/electricity/electricity.csv --model tshm --epochs 13 --dataset_class custom --batch_size 32 --d_model 256 --ff_hidden 256 --n_layers 32 --r 296 --input_len 48 --pred_len 48  --lr 1e-3



















#scp -r yl4971@env-d3qun6i011bs73eb1oa0@ssh-yai.worklink.work:/workspace/tshm_transformer86.py/ F:\tshm






#python3 tshm_transformer70.py --data_dir /workspace/time_serie --model tshm --epochs 7 --dataset_class customs
