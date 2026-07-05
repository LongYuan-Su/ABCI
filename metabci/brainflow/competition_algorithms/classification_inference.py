# -*- coding: utf-8 -*-
"""
Inference script for the first diagnostic ablation final model.

Data:
    009/cupture_data_part2/009_20260620_182128_data.npy
    009/cupture_data_part2/009_20260620_182128_labels.json

Labels:
    code=1 -> rest, label=0
    code=2 -> imagined swallow, label=1

Preprocessing:
    The npy data is already in uV, so this script does not multiply by 1e6.
    EEG: 4-30 Hz
    EMG: 2-200 Hz
    ECG: 0.5-2 Hz

Time-frequency representation:
    This script must match the final training script:
        EEG + EMG
        50 frequency points
        50 ms time pooling, so each sample is [channels, 50, 100]
        JSON timestamp offset = 0 s

Model:
    Each modality uses two branches inspired by the provided model:
        1. Raw time-frequency channel-node readout branch
        2. Local-Global channel graph branch
    Raw and Local-Global branches are always kept.

Input:
    Loads the checkpoint produced by:
        运行_009_时频图_二分支RawLocalGlobal_第一个消融_最终模型构建.py
"""

import copy
import json
import os
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, fftconvolve, sosfilt, sosfiltfilt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# =========================================================
# 0. Config
# =========================================================

@dataclass
class Config:
    data_root: str = r"E:\竞赛\分类\009"
    output_dir: str = r"E:\竞赛\分类\output_009_tfr_raw_localglobal_diagnostic_ablation_5fold"

    sfreq: float = 500.0
    window_sec: float = 5.0
    # Window start = JSON timestamp + window_offset_sec.
    # This tests whether the label timestamp is a cue time rather than the
    # strongest imagined-swallow interval.
    window_offset_sec: float = 0.0
    n_channels: int = 16
    convert_v_to_uv: bool = False

    eeg_idx: tuple = (0, 1, 2, 3, 4, 5, 10, 11, 13)
    emg_idx: tuple = (6, 7, 8, 9, 14, 15)
    ecg_idx: tuple = (12,)

    eeg_filter_low_hz: float = 4.0
    eeg_filter_high_hz: float = 30.0
    emg_filter_low_hz: float = 2.0
    emg_filter_high_hz: float = 200.0
    ecg_filter_low_hz: float = 0.5
    ecg_filter_high_hz: float = 2.0

    # TFR settings. Raw output is [C, freq_points, 2500].
    # 50 points retain the selected bands while reducing redundant frequency features.
    freq_points: int = 50
    morlet_cycles: float = 6.0
    # Average-pool the TFR time axis. 1 keeps all 2500 points;
    # 25 gives 100 points (50 ms bins), 50 gives 50 points (100 ms bins).
    tfr_time_downsample: int = 25
    tfr_log_amplitude: bool = True
    # Do not standardize each individual TFR map. Per-window standardization
    # removes relative band-power and amplitude information that may be useful for EMG.
    tfr_per_channel_standardize: bool = False
    cache_tfr_to_disk: bool = True

    rest_codes: tuple = (1,)
    swallow_codes: tuple = (2,)

    # 只使用标签文件中真实截取到的有效 5s 样本，不自动补充、不做增强、不按 target 数量补齐。
    # 0 表示不限制；若想快速试验每类前 50 个真实样本，设为 50。
    max_samples_per_class: int = 0

    n_splits: int = 5
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42

    # 50-frequency TFR maps permit batch_size=4 on most GPUs. Larger batches
    # make gradients less noisy than batch_size=2 while retaining the full 5 s map.
    batch_size: int = 4
    epochs: int = 150
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 15
    min_epochs_before_early_stop: int = 30
    grad_clip: float = 3.0
    num_workers: int = 0

    # Model settings
    # Both Raw and Local-Global branches are retained. The widths remain modest
    # for the small dataset but are sufficient for a stable training-set fit.
    raw_hidden_dim: int = 32
    graph_hidden_dim: int = 32
    modality_embed_dim: int = 48
    fusion_hidden_dim: int = 64
    dropout: float = 0.15
    classifier_dropout: float = 0.15
    channel_graph_topk: int = 2
    use_abs_corr: bool = True
    channel_nhead: int = 4

    branch_init_raw: float = 1.2
    branch_init_localglobal: float = 1.0
    modality_init_eeg: float = 1.0
    modality_init_emg: float = 1.2
    modality_init_ecg: float = 0.4

    # 消融实验只做窗口、时频维度、模态和小样本诊断；
    # 不消融 Raw / LocalGlobal 两个分支。
    use_eeg_modality: bool = True
    use_emg_modality: bool = True
    use_ecg_modality: bool = True

    # A validation split has only about 32 samples per fold. Keep threshold=0.5
    # during the primary comparison instead of selecting an unstable tiny-set threshold.
    use_val_threshold: bool = False
    threshold_metric: str = "bacc"


# =========================================================
# 1. Utilities and preprocessing
# =========================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_2d_channel_first(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={data.shape}")
    if data.shape[0] <= 128 and data.shape[1] > data.shape[0]:
        return data
    if data.shape[1] <= 128 and data.shape[0] > data.shape[1]:
        return data.T
    return data


def bandpass_filter(data: np.ndarray, sfreq: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.999)
    if low <= 0 or high <= low:
        raise ValueError(f"Invalid band-pass range: {low_hz}-{high_hz} Hz")
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, data, axis=-1)
    except ValueError:
        return sosfilt(sos, data, axis=-1)


def modality_bandpass_filter(data: np.ndarray, cfg: Config) -> np.ndarray:
    out = np.zeros_like(data, dtype=np.float64)
    out[list(cfg.eeg_idx)] = bandpass_filter(data[list(cfg.eeg_idx)], cfg.sfreq, cfg.eeg_filter_low_hz, cfg.eeg_filter_high_hz)
    out[list(cfg.emg_idx)] = bandpass_filter(data[list(cfg.emg_idx)], cfg.sfreq, cfg.emg_filter_low_hz, cfg.emg_filter_high_hz)
    out[list(cfg.ecg_idx)] = bandpass_filter(data[list(cfg.ecg_idx)], cfg.sfreq, cfg.ecg_filter_low_hz, cfg.ecg_filter_high_hz)
    return out


def load_data_array(path: Path, cfg: Config) -> np.ndarray:
    data = ensure_2d_channel_first(np.load(path))
    if data.shape[0] < cfg.n_channels:
        raise ValueError(f"{path} has {data.shape[0]} channels, expected at least {cfg.n_channels}")
    data = data[: cfg.n_channels].astype(np.float64)
    if cfg.convert_v_to_uv:
        data = data * 1e6
    data = modality_bandpass_filter(data, cfg)
    return data


def find_sessions(cfg: Config):
    root = Path(cfg.data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {root.resolve()}")
    sessions = []
    for data_path in sorted(root.rglob("*_data.npy")):
        prefix = data_path.name[:-len("_data.npy")]
        label_path = data_path.with_name(prefix + "_labels.json")
        if not label_path.exists():
            warnings.warn(f"Skip {data_path}: missing labels json")
            continue
        sessions.append((prefix, data_path, label_path))
    if not sessions:
        raise RuntimeError(f"No *_data.npy + *_labels.json pairs found under {root.resolve()}")
    return sessions


def load_events(label_path: Path):
    with open(label_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    if not isinstance(events, list):
        raise ValueError(f"{label_path} should contain a list of events")
    out = []
    for i, ev in enumerate(events):
        if "timestamp_sec" not in ev or "code" not in ev:
            continue
        item = dict(ev)
        item["_event_index"] = i
        item["timestamp_sec"] = float(item["timestamp_sec"])
        item["code"] = int(item["code"])
        out.append(item)
    return sorted(out, key=lambda x: (x["timestamp_sec"], x["_event_index"]))


def code_to_label(code: int, cfg: Config):
    if int(code) in set(cfg.rest_codes):
        return 0, "rest"
    if int(code) in set(cfg.swallow_codes):
        return 1, "imagined_swallow"
    return None, None


def collect_valid_windows(cfg: Config):
    win_samples = int(round(cfg.sfreq * cfg.window_sec))
    offset_samples = int(round(cfg.window_offset_sec * cfg.sfreq))
    rows = []
    for session_id, data_path, label_path in find_sessions(cfg):
        data = load_data_array(data_path, cfg)
        n_samples = data.shape[1]
        duration_sec = n_samples / cfg.sfreq
        for ev in load_events(label_path):
            label, segment = code_to_label(ev["code"], cfg)
            if label is None:
                continue
            start = int(round(ev["timestamp_sec"] * cfg.sfreq)) + offset_samples
            end = start + win_samples
            if start < 0 or end > n_samples:
                warnings.warn(
                    f"Skip event without full 5s window: session={session_id}, "
                    f"event={ev.get('name', '')}, timestamp={ev['timestamp_sec']:.3f}, "
                    f"offset={cfg.window_offset_sec:.3f}s, "
                    f"duration={duration_sec:.3f}"
                )
                continue
            rows.append({
                "x": data[:, start:end].astype(np.float32),
                "y": int(label),
                "segment_type": segment,
                "session_id": session_id,
                "event_index": int(ev["_event_index"]),
                "event_name": str(ev.get("name", "")),
                "timestamp_sec": float(ev["timestamp_sec"]),
                "window_offset_sec": float(cfg.window_offset_sec),
                "window_start_sec": float(start / cfg.sfreq),
                "window_end_sec": float(end / cfg.sfreq),
                "start_sample": int(start),
                "end_sample": int(end),
                "source": str(data_path),
                "is_augmented": False,
            })
    rows.sort(key=lambda r: (r["source"], r["timestamp_sec"], r["event_index"]))
    return rows


def build_window_dataset(cfg: Config):
    """
    构建窗口级数据集。

    关键修改：
    1. 只使用 collect_valid_windows() 从标签事件中真实截取到的完整 5s 窗口。
    2. 不再调用 augment_row()，不再调用 fill_to_count()。
    3. 不再使用 target_rest_samples / target_swallow_samples 自动补齐或截断样本。
    4. 只有 cfg.max_samples_per_class > 0 时，才会对每一类取前 N 个真实样本，用于快速实验。
    """
    rows = collect_valid_windows(cfg)
    rest = [r for r in rows if r["y"] == 0]
    swallow = [r for r in rows if r["y"] == 1]

    if len(rest) == 0 or len(swallow) == 0:
        raise RuntimeError(f"At least one class is empty: rest={len(rest)}, swallow={len(swallow)}")

    if int(getattr(cfg, "max_samples_per_class", 0)) > 0:
        limit = int(cfg.max_samples_per_class)
        rest = rest[:limit]
        swallow = swallow[:limit]

    selected = rest + swallow
    selected.sort(key=lambda r: (r["source"], r["timestamp_sec"], r["event_index"]))

    X = np.stack([r["x"] for r in selected], axis=0).astype(np.float32)
    y = np.asarray([r["y"] for r in selected], dtype=np.int64)
    meta = pd.DataFrame([
        {
            "sample_id": i,
            "label": int(r["y"]),
            "segment_type": r["segment_type"],
            "session_id": r["session_id"],
            "event_index": int(r["event_index"]),
            "event_name": r["event_name"],
            "timestamp_sec": round(float(r["timestamp_sec"]), 6),
            "window_offset_sec": round(float(r["window_offset_sec"]), 6),
            "window_start_sec": round(float(r["window_start_sec"]), 6),
            "window_end_sec": round(float(r["window_end_sec"]), 6),
            "start_sample": int(r["start_sample"]),
            "end_sample": int(r["end_sample"]),
            "is_augmented": False,
            "source": r["source"],
        }
        for i, r in enumerate(selected)
    ])
    return X, y, meta


class RawWindowStandardizer:
    def __init__(self, eps: float = 1e-6, clip_value: float = 8.0):
        self.eps = eps
        self.clip_value = clip_value
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=(0, 2), keepdims=True)
        self.std = X.std(axis=(0, 2), keepdims=True) + self.eps
        return self

    def transform(self, X):
        X = (X - self.mean) / self.std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.clip(X, -self.clip_value, self.clip_value)
        return X.astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "clip_value": self.clip_value}


# =========================================================
# 2. Time-frequency transform
# =========================================================

def modality_frequency_grid(modality: str, cfg: Config) -> np.ndarray:
    if modality == "eeg":
        return np.linspace(cfg.eeg_filter_low_hz, cfg.eeg_filter_high_hz, cfg.freq_points, dtype=np.float64)
    if modality == "emg":
        return np.linspace(cfg.emg_filter_low_hz, cfg.emg_filter_high_hz, cfg.freq_points, dtype=np.float64)
    if modality == "ecg":
        return np.linspace(cfg.ecg_filter_low_hz, cfg.ecg_filter_high_hz, cfg.freq_points, dtype=np.float64)
    raise ValueError(f"Unknown modality: {modality}")


def morlet_wavelet(freq_hz: float, sfreq: float, n_cycles: float, max_len: int) -> np.ndarray:
    sigma_t = n_cycles / (2.0 * np.pi * max(freq_hz, 1e-6))
    half_samples = int(np.ceil(3.0 * sigma_t * sfreq))
    half_samples = max(4, min(half_samples, max_len - 1))
    t = np.arange(-half_samples, half_samples + 1, dtype=np.float64) / sfreq
    wavelet = np.exp(2j * np.pi * freq_hz * t) * np.exp(-(t ** 2) / (2.0 * sigma_t ** 2))
    wavelet = wavelet - wavelet.mean()
    norm = np.sqrt(np.sum(np.abs(wavelet) ** 2)) + 1e-12
    return wavelet / norm


def cwt_morlet_channel(x: np.ndarray, freqs: np.ndarray, cfg: Config) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.empty((len(freqs), len(x)), dtype=np.float32)
    for i, freq in enumerate(freqs):
        wavelet = morlet_wavelet(float(freq), cfg.sfreq, cfg.morlet_cycles, max_len=len(x))
        conv = fftconvolve(x, wavelet.conj()[::-1], mode="same")
        amp = np.abs(conv).astype(np.float32)
        out[i] = amp
    return out


def compute_modality_tfr(window: np.ndarray, modality: str, cfg: Config) -> np.ndarray:
    freqs = modality_frequency_grid(modality, cfg)
    tfr = np.empty((window.shape[0], cfg.freq_points, window.shape[1]), dtype=np.float32)
    for ch in range(window.shape[0]):
        tfr[ch] = cwt_morlet_channel(window[ch], freqs, cfg)

    if cfg.tfr_log_amplitude:
        tfr = np.log1p(tfr)
    downsample = int(getattr(cfg, "tfr_time_downsample", 1))
    if downsample > 1:
        n_time = tfr.shape[-1]
        usable = (n_time // downsample) * downsample
        if usable <= 0:
            raise ValueError(f"Invalid tfr_time_downsample={downsample} for time length={n_time}")
        if usable != n_time:
            tfr = tfr[..., :usable]
        tfr = tfr.reshape(tfr.shape[0], tfr.shape[1], usable // downsample, downsample).mean(axis=-1)
    if cfg.tfr_per_channel_standardize:
        mean = tfr.mean(axis=(1, 2), keepdims=True)
        std = tfr.std(axis=(1, 2), keepdims=True) + 1e-6
        tfr = (tfr - mean) / std
        tfr = np.clip(np.nan_to_num(tfr, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
    return tfr.astype(np.float32)


class TFRWindowDataset(Dataset):
    def __init__(self, X, y, cfg: Config, cache_dir=None, split_name="train"):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.cfg = cfg
        self.split_name = split_name
        self.eeg_idx = list(cfg.eeg_idx)
        self.emg_idx = list(cfg.emg_idx)
        self.ecg_idx = list(cfg.ecg_idx)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.y)

    def _cache_path(self, idx: int) -> Path:
        tag = (
            f"f{self.cfg.freq_points}"
            f"_td{int(getattr(self.cfg, 'tfr_time_downsample', 1))}"
            f"_off{int(round(float(getattr(self.cfg, 'window_offset_sec', 0.0)) * 1000.0))}ms"
            f"_log{int(bool(self.cfg.tfr_log_amplitude))}"
            f"_std{int(bool(self.cfg.tfr_per_channel_standardize))}"
        )
        return self.cache_dir / f"{self.split_name}_{idx:04d}_{tag}.npz"

    def _make_tfr(self, idx: int):
        x = self.X[idx]
        eeg = compute_modality_tfr(x[self.eeg_idx], "eeg", self.cfg)
        emg = compute_modality_tfr(x[self.emg_idx], "emg", self.cfg)
        ecg = compute_modality_tfr(x[self.ecg_idx], "ecg", self.cfg)
        return eeg, emg, ecg

    def __getitem__(self, idx):
        if self.cache_dir is not None:
            path = self._cache_path(idx)
            if path.exists():
                obj = np.load(path)
                eeg = obj["eeg"].astype(np.float32)
                emg = obj["emg"].astype(np.float32)
                ecg = obj["ecg"].astype(np.float32)
            else:
                eeg, emg, ecg = self._make_tfr(idx)
                np.savez_compressed(path, eeg=eeg.astype(np.float16), emg=emg.astype(np.float16), ecg=ecg.astype(np.float16))
        else:
            eeg, emg, ecg = self._make_tfr(idx)
        return (
            torch.tensor(eeg, dtype=torch.float32),
            torch.tensor(emg, dtype=torch.float32),
            torch.tensor(ecg, dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# =========================================================
# 3. Raw + Local-Global model
# =========================================================

def finite_tensor(x, nan=0.0, pos=1e4, neg=-1e4):
    return torch.nan_to_num(x, nan=nan, posinf=pos, neginf=neg)


def normalize_adj_dense(A, eps=1e-8):
    A = finite_tensor(A.float(), nan=0.0, pos=1.0, neg=0.0)
    A = torch.clamp(A, min=0.0, max=1e4)
    deg = A.sum(dim=-1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    return deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)


def channel_corr_from_tfr(x, use_abs=True, topk=2, eps=1e-8):
    # x: [B,C,F,T]
    B, C, Freq, T = x.shape
    # Downsample only for adjacency estimation to keep correlation cheap.
    x_small = F.adaptive_avg_pool2d(x, output_size=(min(32, Freq), min(80, T)))
    series = x_small.reshape(B, C, -1)
    series = series - series.mean(dim=-1, keepdim=True)
    series = series / (series.std(dim=-1, keepdim=True) + eps)
    A = torch.matmul(series, series.transpose(-1, -2)) / max(series.shape[-1] - 1, 1)
    A = finite_tensor(A, nan=0.0, pos=0.0, neg=0.0)
    A = A.abs() if use_abs else torch.clamp(A, min=0.0)
    if topk is not None and 0 < topk < C:
        A_sparse = torch.zeros_like(A)
        eye_mask = torch.eye(C, device=x.device).bool().view(1, C, C)
        A_no_diag = A.masked_fill(eye_mask, -1e9)
        idx = torch.topk(A_no_diag, k=topk, dim=-1).indices
        A_sparse.scatter_(-1, idx, torch.gather(A, dim=-1, index=idx))
        A = torch.maximum(A_sparse, A_sparse.transpose(-1, -2))
    eye = torch.eye(C, device=x.device, dtype=x.dtype).view(1, C, C)
    A = A * (1.0 - eye) + eye
    return A


class TFRNodeCNNEncoder(nn.Module):
    """Shared channel-node encoder for [F,T] maps."""
    def __init__(self, hidden_dim=48, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 25), stride=(1, 5), padding=(2, 12), bias=False),
            # GroupNorm does not depend on the small DataLoader batch size and
            # is therefore more stable than BatchNorm for this dataset.
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.Conv2d(8, 16, kernel_size=(5, 9), stride=(1, 2), padding=(2, 4), bias=False),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((10, 20)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(16 * 10 * 20, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B,C,F,T]
        B, C, Freq, T = x.shape
        h = self.net(x.reshape(B * C, 1, Freq, T))
        return h.reshape(B, C, -1)


class RawTFRChannelReadoutBranch(nn.Module):
    """Raw branch: direct channel-node readout from time-frequency maps."""
    def __init__(self, hidden_dim=48, dropout=0.25):
        super().__init__()
        self.node_encoder = TFRNodeCNNEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.channel_att = nn.Sequential(
            nn.Linear(hidden_dim, max(8, hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(8, hidden_dim // 2), 1),
        )
        self.out_dim = hidden_dim * 3

    def forward(self, x):
        h = self.node_encoder(x)
        score = self.channel_att(h).squeeze(-1)
        weight = torch.softmax(score, dim=1).unsqueeze(-1)
        att_pool = torch.sum(h * weight, dim=1)
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        return torch.cat([att_pool, mean_pool, max_pool], dim=-1)


class LocalGlobalGraphFilterLayer(nn.Module):
    def __init__(self, d_model, dropout=0.25):
        super().__init__()
        self.lin_self = nn.Linear(d_model, d_model)
        self.lin_neigh = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, A):
        A_norm = normalize_adj_dense(A)
        neigh = torch.bmm(A_norm, h)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        out = F.relu(self.norm(out))
        return self.dropout(out)


class LocalGlobalBlock(nn.Module):
    """Local graph filter + global self-attention, same spirit as the provided code."""
    def __init__(self, d_model=48, nhead=4, dropout=0.25):
        super().__init__()
        if d_model % nhead != 0:
            nhead = 1
        self.local_filter = LocalGlobalGraphFilterLayer(d_model, dropout=dropout)
        self.global_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, h, A):
        h = finite_tensor(h)
        A = finite_tensor(A, nan=0.0, pos=1.0, neg=0.0)
        h_local = self.local_filter(h, A)
        h_global, _ = self.global_attn(h, h, h, need_weights=False)
        h_global = finite_tensor(h_global)
        h_fuse = self.fusion(torch.cat([h, h_local, h_global], dim=-1))
        return self.out_norm(h + h_fuse)


class LocalGlobalTFRChannelBranch(nn.Module):
    """Local-Global channel branch from channel correlation graph."""
    def __init__(self, num_channels, hidden_dim=48, dropout=0.25, topk=2, use_abs_corr=True, nhead=4):
        super().__init__()
        self.num_channels = int(num_channels)
        self.topk = topk
        self.use_abs_corr = bool(use_abs_corr)
        self.node_encoder = TFRNodeCNNEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.channel_embedding = nn.Embedding(num_channels, hidden_dim)
        self.block = LocalGlobalBlock(hidden_dim, nhead=nhead, dropout=dropout)
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        B, C, _, _ = x.shape
        A = channel_corr_from_tfr(x, use_abs=self.use_abs_corr, topk=self.topk)
        h = self.node_encoder(x)
        ids = torch.arange(C, device=x.device)
        h = h + self.channel_embedding(ids).unsqueeze(0)
        h = self.block(h, A)
        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1).values
        return torch.cat([mean_pool, max_pool], dim=-1)


class TwoBranchGatedFusion(nn.Module):
    def __init__(self, raw_dim, lg_dim, out_dim=80, dropout=0.25, init=(1.2, 1.0)):
        super().__init__()
        self.raw_proj = nn.Sequential(nn.Linear(raw_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lg_proj = nn.Sequential(nn.Linear(lg_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU(), nn.Dropout(dropout))
        self.branch_logits = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        self.out_dim = out_dim

    def forward(self, raw_emb, lg_emb):
        raw = self.raw_proj(finite_tensor(raw_emb))
        lg = self.lg_proj(finite_tensor(lg_emb))
        w = torch.softmax(finite_tensor(self.branch_logits, nan=0.0, pos=5.0, neg=-5.0), dim=0)
        return finite_tensor(w[0] * raw + w[1] * lg), w


class ModalityTFTwoBranchEncoder(nn.Module):
    def __init__(self, num_channels, cfg: Config):
        super().__init__()
        self.raw_branch = RawTFRChannelReadoutBranch(cfg.raw_hidden_dim, cfg.dropout)
        self.localglobal_branch = LocalGlobalTFRChannelBranch(
            num_channels=num_channels,
            hidden_dim=cfg.graph_hidden_dim,
            dropout=cfg.dropout,
            topk=cfg.channel_graph_topk,
            use_abs_corr=cfg.use_abs_corr,
            nhead=cfg.channel_nhead,
        )
        self.fusion = TwoBranchGatedFusion(
            raw_dim=self.raw_branch.out_dim,
            lg_dim=self.localglobal_branch.out_dim,
            out_dim=cfg.modality_embed_dim,
            dropout=cfg.dropout,
            init=(cfg.branch_init_raw, cfg.branch_init_localglobal),
        )
        self.out_dim = cfg.modality_embed_dim

    def forward(self, x):
        raw_emb = self.raw_branch(x)
        lg_emb = self.localglobal_branch(x)
        return self.fusion(raw_emb, lg_emb)


class ModalityGatedFusion(nn.Module):
    def __init__(self, embed_dim, init=(1.0, 1.2, 0.4), active=(True, True, True)):
        super().__init__()
        self.modality_logits = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        mask = torch.tensor(active, dtype=torch.bool)
        if not bool(mask.any()):
            raise ValueError("EEG/EMG/ECG 不能全部关闭。")
        self.register_buffer("active_mask", mask)
        self.out_dim = embed_dim

    def forward(self, eeg_emb, emg_emb, ecg_emb):
        logits = finite_tensor(self.modality_logits, nan=0.0, pos=5.0, neg=-5.0)
        logits = logits.masked_fill(~self.active_mask, -1e4)
        w = torch.softmax(logits, dim=0)
        fused = w[0] * eeg_emb + w[1] * emg_emb + w[2] * ecg_emb
        return finite_tensor(fused), w


class TimeFrequencyRawLocalGlobalNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.eeg_encoder = ModalityTFTwoBranchEncoder(len(cfg.eeg_idx), cfg)
        self.emg_encoder = ModalityTFTwoBranchEncoder(len(cfg.emg_idx), cfg)
        self.ecg_encoder = ModalityTFTwoBranchEncoder(len(cfg.ecg_idx), cfg)
        self.modality_fusion = ModalityGatedFusion(
            cfg.modality_embed_dim,
            init=(cfg.modality_init_eeg, cfg.modality_init_emg, cfg.modality_init_ecg),
            active=(cfg.use_eeg_modality, cfg.use_emg_modality, cfg.use_ecg_modality),
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.modality_embed_dim, cfg.fusion_hidden_dim),
            nn.LayerNorm(cfg.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim // 2),
            nn.LayerNorm(cfg.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(cfg.classifier_dropout),
            nn.Linear(cfg.fusion_hidden_dim // 2, 1),
        )

    def forward(self, eeg, emg, ecg, return_aux=False):
        B = eeg.size(0)
        D = self.cfg.modality_embed_dim
        device = eeg.device
        dtype = eeg.dtype
        if self.cfg.use_eeg_modality:
            eeg_emb, eeg_bw = self.eeg_encoder(eeg)
        else:
            eeg_emb = torch.zeros(B, D, device=device, dtype=dtype)
            eeg_bw = torch.zeros(2, device=device, dtype=dtype)
        if self.cfg.use_emg_modality:
            emg_emb, emg_bw = self.emg_encoder(emg)
        else:
            emg_emb = torch.zeros(B, D, device=device, dtype=dtype)
            emg_bw = torch.zeros(2, device=device, dtype=dtype)
        if self.cfg.use_ecg_modality:
            ecg_emb, ecg_bw = self.ecg_encoder(ecg)
        else:
            ecg_emb = torch.zeros(B, D, device=device, dtype=dtype)
            ecg_bw = torch.zeros(2, device=device, dtype=dtype)
        fused, modality_w = self.modality_fusion(eeg_emb, emg_emb, ecg_emb)
        logit = self.classifier(fused).squeeze(1)
        logit = torch.nan_to_num(logit, nan=0.0, posinf=20.0, neginf=-20.0)
        if return_aux:
            return logit, {
                "modality_weight": modality_w.detach(),
                "eeg_branch_weight": eeg_bw.detach(),
                "emg_branch_weight": emg_bw.detach(),
                "ecg_branch_weight": ecg_bw.detach(),
            }
        return logit


# =========================================================
# 4. Training and evaluation
# =========================================================

def metrics_from_prob(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    except Exception:
        auc = np.nan
    auc_value = float(auc) if np.isfinite(auc) else np.nan
    hybrid = 0.5 * float(bacc) + 0.5 * auc_value if np.isfinite(auc_value) else float(bacc)
    return {
        "acc": float(acc),
        "bacc": float(bacc),
        "f1": float(f1),
        "auc": auc_value,
        "hybrid": float(hybrid),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }, y_pred


def find_best_threshold(y_true, y_prob, metric="bacc"):
    best_thr, best_score = 0.5, -np.inf
    for thr in np.linspace(0.05, 0.95, 181):
        metrics, _ = metrics_from_prob(y_true, y_prob, threshold=float(thr))
        score = metrics[metric] if metric in {"acc", "bacc", "f1"} else metrics["bacc"]
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def run_epoch(model, loader, optimizer, criterion, device, train=True, grad_clip=3.0):
    model.train(train)
    total_loss, n_seen = 0.0, 0
    y_true, y_prob = [], []
    modality_weights, eeg_bw, emg_bw, ecg_bw = [], [], [], []

    for eeg, emg, ecg, y in loader:
        eeg = eeg.to(device, non_blocking=True)
        emg = emg.to(device, non_blocking=True)
        ecg = ecg.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits, aux = model(eeg, emg, ecg, return_aux=True)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob.extend(prob)
        y_true.extend(y.detach().cpu().numpy())
        total_loss += float(loss.item()) * y.size(0)
        n_seen += y.size(0)
        modality_weights.append(aux["modality_weight"].detach().cpu().numpy())
        eeg_bw.append(aux["eeg_branch_weight"].detach().cpu().numpy())
        emg_bw.append(aux["emg_branch_weight"].detach().cpu().numpy())
        ecg_bw.append(aux["ecg_branch_weight"].detach().cpu().numpy())

    avg_loss = total_loss / max(n_seen, 1)
    metrics, _ = metrics_from_prob(y_true, y_prob, threshold=0.5)
    aux_mean = {
        "modality_weight": np.stack(modality_weights).mean(axis=0),
        "eeg_branch_weight": np.stack(eeg_bw).mean(axis=0),
        "emg_branch_weight": np.stack(emg_bw).mean(axis=0),
        "ecg_branch_weight": np.stack(ecg_bw).mean(axis=0),
    }
    return avg_loss, np.asarray(y_true).astype(int), np.asarray(y_prob).astype(float), metrics, aux_mean


def split_indices(y, cfg: Config):
    idx = np.arange(len(y))
    trainval_idx, test_idx = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=cfg.val_size,
        random_state=cfg.random_state,
        stratify=y[trainval_idx],
    )
    return train_idx, val_idx, test_idx


def get_ablation_settings():
    return [
        {
            "name": "eeg_emg_pool50ms_offset0",
            "description": "主诊断：EEG+EMG，50 频点，TFR 时间轴 50ms 平均池化，窗口从 JSON 时间戳开始。",
            "overrides": {
                "use_eeg_modality": True,
                "use_emg_modality": True,
                "use_ecg_modality": False,
                "freq_points": 50,
                "tfr_time_downsample": 25,
                "window_offset_sec": 0.0,
            },
        },
        {
            "name": "eeg_emg_pool50ms_offset0p5",
            "description": "窗口偏移诊断：EEG+EMG，窗口整体后移 0.5s，检验 JSON 时间戳是否偏早。",
            "overrides": {
                "use_eeg_modality": True,
                "use_emg_modality": True,
                "use_ecg_modality": False,
                "freq_points": 50,
                "tfr_time_downsample": 25,
                "window_offset_sec": 0.5,
            },
        },
        {
            "name": "eeg_emg_pool50ms_offset1p0",
            "description": "窗口偏移诊断：EEG+EMG，窗口整体后移 1.0s，检验想象吞咽反应延迟。",
            "overrides": {
                "use_eeg_modality": True,
                "use_emg_modality": True,
                "use_ecg_modality": False,
                "freq_points": 50,
                "tfr_time_downsample": 25,
                "window_offset_sec": 1.0,
            },
        },
        {
            "name": "eeg_emg_pool100ms_offset0",
            "description": "降维诊断：EEG+EMG，TFR 时间轴 100ms 平均池化，进一步降低 5s 输入复杂度。",
            "overrides": {
                "use_eeg_modality": True,
                "use_emg_modality": True,
                "use_ecg_modality": False,
                "freq_points": 50,
                "tfr_time_downsample": 50,
                "window_offset_sec": 0.0,
            },
        },
        {
            "name": "eeg_emg_keep2500_offset0",
            "description": "复杂度对照：EEG+EMG，保留全部 2500 个时间点，用来对比时间降采样是否改善泛化。",
            "overrides": {
                "use_eeg_modality": True,
                "use_emg_modality": True,
                "use_ecg_modality": False,
                "freq_points": 50,
                "tfr_time_downsample": 1,
                "window_offset_sec": 0.0,
            },
        },
        {
            "name": "eeg_emg_freq100_pool50ms",
            "description": "频率点诊断：EEG+EMG，100 频点 + 50ms 时间池化，检验 50 频点是否丢失信息。",
            "overrides": {
                "use_eeg_modality": True,
                "use_emg_modality": True,
                "use_ecg_modality": False,
                "freq_points": 100,
                "tfr_time_downsample": 25,
                "window_offset_sec": 0.0,
            },
        },
        {
            "name": "eeg_only_pool50ms",
            "description": "模态诊断：仅 EEG，保留 Raw+LocalGlobal 两分支，判断脑电单模态是否有稳定信息。",
            "overrides": {
                "use_eeg_modality": True,
                "use_emg_modality": False,
                "use_ecg_modality": False,
                "freq_points": 50,
                "tfr_time_downsample": 25,
                "window_offset_sec": 0.0,
            },
        },
        {
            "name": "emg_only_pool50ms",
            "description": "模态诊断：仅 EMG，保留 Raw+LocalGlobal 两分支，判断肌电是否有可分信息。",
            "overrides": {
                "use_eeg_modality": False,
                "use_emg_modality": True,
                "use_ecg_modality": False,
                "freq_points": 50,
                "tfr_time_downsample": 25,
                "window_offset_sec": 0.0,
            },
        },
        {
            "name": "full_pool50ms",
            "description": "模态诊断：EEG+EMG+ECG 全模态，检验 ECG 是否带来增益或噪声。",
            "overrides": {
                "use_eeg_modality": True,
                "use_emg_modality": True,
                "use_ecg_modality": True,
                "freq_points": 50,
                "tfr_time_downsample": 25,
                "window_offset_sec": 0.0,
            },
        },
        {
            "name": "first20_per_class_low_regularization",
            "description": "小样本记忆诊断：每类前 20 个真实样本，关闭 dropout/weight_decay，检查模型能否稳定拟合训练数据。",
            "overrides": {
                "use_eeg_modality": True,
                "use_emg_modality": True,
                "use_ecg_modality": False,
                "max_samples_per_class": 20,
                "freq_points": 50,
                "tfr_time_downsample": 25,
                "window_offset_sec": 0.0,
                "dropout": 0.0,
                "classifier_dropout": 0.0,
                "weight_decay": 0.0,
                "lr": 5e-4,
                "epochs": 120,
                "patience": 25,
                "min_epochs_before_early_stop": 40,
            },
        },
    ]


# 默认先运行最关键的 4 组诊断。设置为 () 时运行上面全部实验。
RUN_ABLATION_NAMES = (
    "eeg_emg_pool50ms_offset0",
    "eeg_emg_pool50ms_offset0p5",
    "eeg_emg_pool50ms_offset1p0",
    "eeg_emg_pool100ms_offset0",
)
# RUN_ABLATION_NAMES = (
#     "eeg_emg_pool50ms_offset0",
#     "eeg_emg_pool50ms_offset0p5",
#     "eeg_emg_pool50ms_offset1p0",
#     "eeg_emg_pool100ms_offset0",
#     "eeg_emg_keep2500_offset0",
#     "eeg_emg_freq100_pool50ms",
#     "eeg_only_pool50ms",
#     "emg_only_pool50ms",
#     "full_pool50ms",
#     "first20_per_class_low_regularization",
# )

def apply_overrides(cfg, overrides):
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Config has no field named {key}")
        setattr(cfg, key, value)
    return cfg


def train_one_fold_for_ablation(X, y, meta, train_idx, val_idx, test_idx, fold, cfg, setting):
    fold_dir = Path(cfg.output_dir) / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print(f"{setting['name']} | Fold {fold}/{cfg.n_splits}")
    print("=" * 90)
    print("train label counts:", dict(pd.Series(y[train_idx]).value_counts().sort_index()))
    print("val label counts:", dict(pd.Series(y[val_idx]).value_counts().sort_index()))
    print("test label counts:", dict(pd.Series(y[test_idx]).value_counts().sort_index()))

    standardizer = RawWindowStandardizer().fit(X[train_idx])
    X_std = standardizer.transform(X)

    tfr_cache_dir = (fold_dir / "tfr_cache") if cfg.cache_tfr_to_disk else None
    train_ds = TFRWindowDataset(X_std[train_idx], y[train_idx], cfg, cache_dir=tfr_cache_dir, split_name="train")
    val_ds = TFRWindowDataset(X_std[val_idx], y[val_idx], cfg, cache_dir=tfr_cache_dir, split_name="val")
    test_ds = TFRWindowDataset(X_std[test_idx], y[test_idx], cfg, cache_dir=tfr_cache_dir, split_name="test")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    if fold == 1:
        sample_eeg, sample_emg, sample_ecg, _ = train_ds[0]
        print("TFR shapes per sample:")
        print("  EEG:", tuple(sample_eeg.shape))
        print("  EMG:", tuple(sample_emg.shape))
        print("  ECG:", tuple(sample_ecg.shape))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeFrequencyRawLocalGlobalNet(cfg).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_trainable:,}")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    best_state = None
    best_val_loss = np.inf
    best_epoch = 0
    best_threshold = 0.5
    wait = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_true, train_prob, train_metrics, _ = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True, grad_clip=cfg.grad_clip
        )
        val_loss, val_true, val_prob, val_metrics_05, val_aux = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )
        scheduler.step(val_loss)

        if cfg.use_val_threshold:
            val_thr, _ = find_best_threshold(val_true, val_prob, metric=cfg.threshold_metric)
        else:
            val_thr = 0.5
        val_metrics, _ = metrics_from_prob(val_true, val_prob, threshold=val_thr)

        mw = val_aux["modality_weight"]
        ebw = val_aux["eeg_branch_weight"]
        mbw = val_aux["emg_branch_weight"]
        cbw = val_aux["ecg_branch_weight"]
        history.append({
            "fold": fold,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_loss": val_loss,
            "val_threshold": val_thr,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "modality_weight_eeg": float(mw[0]),
            "modality_weight_emg": float(mw[1]),
            "modality_weight_ecg": float(mw[2]),
            "eeg_branch_raw": float(ebw[0]),
            "eeg_branch_localglobal": float(ebw[1]),
            "emg_branch_raw": float(mbw[0]),
            "emg_branch_localglobal": float(mbw[1]),
            "ecg_branch_raw": float(cbw[0]),
            "ecg_branch_localglobal": float(cbw[1]),
        })

        print(
            f"[{setting['name']} | Fold {fold:02d} | Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_bacc={train_metrics['bacc']:.4f} "
            f"val_loss={val_loss:.4f} val_bacc={val_metrics['bacc']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            best_threshold = float(val_thr)
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if epoch >= cfg.min_epochs_before_early_stop and wait >= cfg.patience:
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}")
                break

    pd.DataFrame(history).to_csv(fold_dir / "fold_history.csv", index=False, encoding="utf-8-sig")
    if best_state is None:
        raise RuntimeError(f"{setting['name']} fold {fold} did not produce a valid best_state.")

    model.load_state_dict(best_state)
    test_loss, test_true, test_prob, test_metrics_05, test_aux = run_epoch(model, test_loader, optimizer, criterion, device, train=False)
    test_metrics, test_pred = metrics_from_prob(test_true, test_prob, threshold=best_threshold)

    # =========================
    # 输出当前 Fold 的测试集指标
    # =========================
    print("\n" + "-" * 90)
    print(f"{setting['name']} | Fold {fold:02d} 测试集结果")
    print("-" * 90)
    print(f"Best Epoch       : {best_epoch}")
    print(f"Best Val Loss    : {best_val_loss:.6f}")
    print(f"Best Threshold   : {best_threshold:.4f}")
    print(f"Test Loss        : {test_loss:.6f}")
    print(f"Test ACC         : {test_metrics['acc']:.4f}")
    print(f"Test BACC        : {test_metrics['bacc']:.4f}")
    print(f"Test F1          : {test_metrics['f1']:.4f}")
    print(f"Test AUC         : {test_metrics['auc']:.4f}")
    print(f"Test Sensitivity : {test_metrics['sensitivity']:.4f}")
    print(f"Test Specificity : {test_metrics['specificity']:.4f}")
    print(
        f"Confusion Matrix : "
        f"TN={test_metrics['tn']}, FP={test_metrics['fp']}, "
        f"FN={test_metrics['fn']}, TP={test_metrics['tp']}"
    )
    print("-" * 90)

    pred_df = pd.DataFrame({"y_true": test_true, "y_prob": test_prob, "y_pred": test_pred})
    pred_df.to_csv(fold_dir / "fold_predictions.csv", index=False, encoding="utf-8-sig")

    result = {
        "name": setting["name"],
        "description": setting["description"],
        "fold": int(fold),
        "test_loss": float(test_loss),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_threshold": float(best_threshold),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        **test_metrics,
        "overrides": json.dumps(setting["overrides"], ensure_ascii=False),
        "modality_weight_eeg": float(test_aux["modality_weight"][0]),
        "modality_weight_emg": float(test_aux["modality_weight"][1]),
        "modality_weight_ecg": float(test_aux["modality_weight"][2]),
        "eeg_branch_raw": float(test_aux["eeg_branch_weight"][0]),
        "eeg_branch_localglobal": float(test_aux["eeg_branch_weight"][1]),
        "emg_branch_raw": float(test_aux["emg_branch_weight"][0]),
        "emg_branch_localglobal": float(test_aux["emg_branch_weight"][1]),
        "ecg_branch_raw": float(test_aux["ecg_branch_weight"][0]),
        "ecg_branch_localglobal": float(test_aux["ecg_branch_weight"][1]),
    }
    pd.DataFrame([result]).to_csv(fold_dir / "fold_test_metrics.csv", index=False, encoding="utf-8-sig")
    return result


def run_one_ablation(setting, root_output_dir):
    cfg = Config()
    cfg.output_dir = str(Path(root_output_dir) / setting["name"])
    cfg = apply_overrides(cfg, setting["overrides"])
    set_seed(cfg.random_state)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    with open(out_dir / "ablation_description.json", "w", encoding="utf-8") as f:
        json.dump(setting, f, ensure_ascii=False, indent=2)

    print("\n" + "#" * 100)
    print("开始五折消融实验:", setting["name"])
    print(setting["description"])
    print("#" * 100)

    X, y, meta = build_window_dataset(cfg)
    print("X raw windows:", X.shape)
    print("label counts:", dict(pd.Series(y).value_counts().sort_index()))
    print("augmented counts:", meta["is_augmented"].value_counts().to_dict(), "(已关闭样本自动补充/增强)")

    cache_dir = out_dir / "raw_window_dataset"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "X_16x2500_modality_filtered_uV.npy", X)
    np.save(cache_dir / "y.npy", y)
    meta.to_csv(cache_dir / "metadata_windows.csv", index=False, encoding="utf-8-sig")
    meta.groupby(["segment_type", "label"]).size().reset_index(name="count").to_csv(
        cache_dir / "segment_counts.csv", index=False, encoding="utf-8-sig"
    )

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    fold_results = []
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        set_seed(cfg.random_state + fold)
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=cfg.val_size,
            stratify=y[trainval_idx],
            random_state=cfg.random_state + fold,
        )
        fold_result = train_one_fold_for_ablation(X, y, meta, train_idx, val_idx, test_idx, fold, cfg, setting)
        fold_results.append(fold_result)
        pd.DataFrame(fold_results).to_csv(out_dir / "classification_fold_results_running.csv", index=False, encoding="utf-8-sig")

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(out_dir / "classification_fold_results.csv", index=False, encoding="utf-8-sig")

    metric_cols = ["acc", "bacc", "f1", "auc", "sensitivity", "specificity", "hybrid"]
    summary = {
        "name": setting["name"],
        "description": setting["description"],
        "overrides": json.dumps(setting["overrides"], ensure_ascii=False),
        "window_offset_sec": float(cfg.window_offset_sec),
        "freq_points": int(cfg.freq_points),
        "tfr_time_downsample": int(cfg.tfr_time_downsample),
        "tfr_time_points": int(round(cfg.sfreq * cfg.window_sec)) // int(cfg.tfr_time_downsample),
        "max_samples_per_class": int(cfg.max_samples_per_class),
        "use_eeg_modality": bool(cfg.use_eeg_modality),
        "use_emg_modality": bool(cfg.use_emg_modality),
        "use_ecg_modality": bool(cfg.use_ecg_modality),
        "dropout": float(cfg.dropout),
        "classifier_dropout": float(cfg.classifier_dropout),
        "weight_decay": float(cfg.weight_decay),
    }
    for col in metric_cols:
        summary[f"{col}_mean"] = float(fold_df[col].mean())
        summary[f"{col}_std"] = float(fold_df[col].std())
    for col in [
        "modality_weight_eeg", "modality_weight_emg", "modality_weight_ecg",
        "eeg_branch_raw", "eeg_branch_localglobal",
        "emg_branch_raw", "emg_branch_localglobal",
        "ecg_branch_raw", "ecg_branch_localglobal",
    ]:
        summary[f"{col}_mean"] = float(fold_df[col].mean())
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / "classification_summary.csv", index=False, encoding="utf-8-sig")

    # =========================
    # 输出当前消融实验的 5-fold 测试集均值 ± 标准差
    # =========================
    print("\n" + "=" * 100)
    print(f"{setting['name']} | 5-fold 测试集汇总")
    print("=" * 100)
    print(f"ACC         : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    print(f"BACC        : {summary['bacc_mean']:.4f} ± {summary['bacc_std']:.4f}")
    print(f"F1          : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print(f"AUC         : {summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}")
    print(f"Sensitivity : {summary['sensitivity_mean']:.4f} ± {summary['sensitivity_std']:.4f}")
    print(f"Specificity : {summary['specificity_mean']:.4f} ± {summary['specificity_std']:.4f}")
    print(f"Hybrid      : {summary['hybrid_mean']:.4f} ± {summary['hybrid_std']:.4f}")
    print("=" * 100)
    return summary

def main():
    root_output_dir = "output_009_tfr_raw_localglobal_diagnostic_ablation_5fold"
    root = Path(root_output_dir)
    root.mkdir(parents=True, exist_ok=True)

    settings = get_ablation_settings()
    if RUN_ABLATION_NAMES:
        keep = set(RUN_ABLATION_NAMES)
        settings = [s for s in settings if s["name"] in keep]

    all_results = []
    failed = []
    for setting in settings:
        try:
            result = run_one_ablation(setting, root_output_dir)
            all_results.append(result)
            pd.DataFrame(all_results).to_csv(root / "ablation_summary_running.csv", index=False, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[失败] {setting['name']}: {exc}")
            failed.append({"name": setting["name"], "description": setting["description"], "error": repr(exc)})
            pd.DataFrame(failed).to_csv(root / "ablation_failed.csv", index=False, encoding="utf-8-sig")

    if all_results:
        summary_df = pd.DataFrame(all_results).sort_values("bacc_mean", ascending=False)
        summary_df.to_csv(root / "ablation_summary.csv", index=False, encoding="utf-8-sig")
        print("\n" + "=" * 100)
        print("消融实验汇总")
        print("=" * 100)
        display_cols = [
            c for c in [
                "name",
                "acc_mean", "acc_std",
                "bacc_mean", "bacc_std",
                "f1_mean", "f1_std",
                "auc_mean", "auc_std",
                "sensitivity_mean", "sensitivity_std",
                "specificity_mean", "specificity_std",
                "hybrid_mean", "hybrid_std",
                "description",
            ]
            if c in summary_df.columns
        ]
        print(summary_df[display_cols].to_string(index=False))

    if failed:
        print("有实验失败，详情见:", root / "ablation_failed.csv")


MODEL_PATH = r"output_009_tfr_raw_localglobal_first_ablation_final_model\final_model.pt"
PREDICT_DATA_ROOT = None
INFERENCE_OUTPUT_DIR = r"output_009_tfr_raw_localglobal_first_ablation_inference"


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def cfg_from_checkpoint(checkpoint) -> Config:
    cfg = Config()
    saved_cfg = checkpoint.get("cfg", {})
    for key, value in saved_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    if PREDICT_DATA_ROOT:
        cfg.data_root = PREDICT_DATA_ROOT
    cfg.output_dir = INFERENCE_OUTPUT_DIR
    cfg.cache_tfr_to_disk = True
    return cfg


def standardizer_from_state(state):
    std = RawWindowStandardizer(clip_value=float(state.get("clip_value", 8.0)))
    std.mean = np.asarray(state["mean"], dtype=np.float32)
    std.std = np.asarray(state["std"], dtype=np.float32)
    return std


def predict_loader(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    modality_weights, eeg_bw, emg_bw, ecg_bw = [], [], [], []
    with torch.no_grad():
        for eeg, emg, ecg, y in loader:
            eeg = eeg.to(device, non_blocking=True)
            emg = emg.to(device, non_blocking=True)
            ecg = ecg.to(device, non_blocking=True)
            logits, aux = model(eeg, emg, ecg, return_aux=True)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            y_prob.extend(prob)
            y_true.extend(y.detach().cpu().numpy())
            modality_weights.append(aux["modality_weight"].detach().cpu().numpy())
            eeg_bw.append(aux["eeg_branch_weight"].detach().cpu().numpy())
            emg_bw.append(aux["emg_branch_weight"].detach().cpu().numpy())
            ecg_bw.append(aux["ecg_branch_weight"].detach().cpu().numpy())
    aux_mean = {
        "modality_weight": np.stack(modality_weights).mean(axis=0),
        "eeg_branch_weight": np.stack(eeg_bw).mean(axis=0),
        "emg_branch_weight": np.stack(emg_bw).mean(axis=0),
        "ecg_branch_weight": np.stack(ecg_bw).mean(axis=0),
    }
    return np.asarray(y_true).astype(int), np.asarray(y_prob).astype(float), aux_mean


def main_inference():
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            f"Please run 运行_009_时频图_二分支RawLocalGlobal_第一个消融_最终模型构建.py first."
        )

    checkpoint = safe_torch_load(model_path, map_location="cpu")
    cfg = cfg_from_checkpoint(checkpoint)
    threshold = float(checkpoint.get("threshold", 0.5))

    out_dir = Path(INFERENCE_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("First ablation final model inference")
    print("=" * 100)
    print("Model path:", model_path)
    print("Data root:", cfg.data_root)
    print("Threshold:", threshold)

    X, y, meta = build_window_dataset(cfg)
    print("X raw windows:", X.shape)
    print("label counts:", dict(pd.Series(y).value_counts().sort_index()))
    print("augmented counts:", meta["is_augmented"].value_counts().to_dict(), "(no augmentation)")

    standardizer = standardizer_from_state(checkpoint["standardizer_state"])
    X_std = standardizer.transform(X)

    tfr_cache_dir = out_dir / "tfr_cache"
    ds = TFRWindowDataset(X_std, y, cfg, cache_dir=tfr_cache_dir, split_name="predict")
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    sample_eeg, sample_emg, sample_ecg, _ = ds[0]
    print("TFR shapes per sample:")
    print("  EEG:", tuple(sample_eeg.shape))
    print("  EMG:", tuple(sample_emg.shape))
    print("  ECG:", tuple(sample_ecg.shape))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeFrequencyRawLocalGlobalNet(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    y_true, y_prob, aux = predict_loader(model, loader, device)
    metrics, y_pred = metrics_from_prob(y_true, y_prob, threshold=threshold)

    pred_df = meta.copy()
    pred_df["y_true"] = y_true
    pred_df["y_prob"] = y_prob
    pred_df["y_pred"] = y_pred
    pred_df["pred_name"] = np.where(pred_df["y_pred"].values == 1, "imagined_swallow", "rest")
    pred_df["is_correct"] = pred_df["y_true"].values == pred_df["y_pred"].values
    pred_csv = out_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")

    summary = {
        "model_path": str(model_path),
        "data_root": str(cfg.data_root),
        "threshold": float(threshold),
        "n_predicted": int(len(pred_df)),
        "n_rest": int((y_true == 0).sum()),
        "n_imagined_swallow": int((y_true == 1).sum()),
        **metrics,
        "modality_weight_eeg": float(aux["modality_weight"][0]),
        "modality_weight_emg": float(aux["modality_weight"][1]),
        "modality_weight_ecg": float(aux["modality_weight"][2]),
        "eeg_branch_raw": float(aux["eeg_branch_weight"][0]),
        "eeg_branch_localglobal": float(aux["eeg_branch_weight"][1]),
        "emg_branch_raw": float(aux["emg_branch_weight"][0]),
        "emg_branch_localglobal": float(aux["emg_branch_weight"][1]),
        "ecg_branch_raw": float(aux["ecg_branch_weight"][0]),
        "ecg_branch_localglobal": float(aux["ecg_branch_weight"][1]),
    }
    pd.DataFrame([summary]).to_csv(out_dir / "prediction_summary.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nMetrics:")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("Predictions saved:", pred_csv)


if __name__ == "__main__":
    main_inference()
