# -*- coding: utf-8 -*-
"""
Run the 009 warm-prior regression experiment.

This script calls warm_prior_score_model.py to build the model, then:
    1. reads 009 raw data and labels,
    2. builds warm-task and imagined-prior paired windows,
    3. converts them to Morlet TFR,
    4. fits the target 1-point score from 009_subject_score_target.csv,
    5. writes one compact CSV result.
"""

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt, sosfiltfilt

import torch
import torch.nn as nn

from warm_prior_score_model import ModelConfig, build_model


@dataclass
class ExperimentConfig:
    data_dir: str = r"e:\竞赛\量化\009"
    target_csv: str = r"e:\竞赛\量化\009_subject_score_target.csv"
    output_dir: str = r"e:\竞赛\量化\009_warm_prior_regression_results"
    subject_id: str = "009"

    sfreq: float = 500.0
    window_sec: float = 5.0
    n_channels: int = 16
    n_freqs: int = 100
    n_times: int = 2500
    n_tasks: int = 10
    data_unit: str = "uV"

    eeg_idx: tuple = (0, 1, 2, 3, 4, 5, 10, 11, 13)
    emg_idx: tuple = (6, 7, 8, 9, 14, 15)
    ecg_idx: tuple = (12,)
    eeg_band: tuple = (4.0, 30.0)
    emg_band: tuple = (2.0, 200.0)
    ecg_band: tuple = (0.5, 2.0)
    filter_order: int = 4
    morlet_cycles: float = 6.0
    tf_log_amplitude: bool = True
    tf_per_channel_standardize: bool = True
    cache_time_frequency: bool = True

    imagined_code: int = 2
    warm_code: int = 4
    prior_lambda: float = 0.04

    raw_hidden_dim: int = 48
    task_embed_dim: int = 80
    task_att_hidden: int = 64
    dropout: float = 0.25
    task_mean_residual_ratio: float = 0.20

    epochs: int = 120
    lr: float = 3e-4
    weight_decay: float = 3e-4
    grad_clip: float = 1.0
    random_state: int = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_target_csv(path, subject_id, score=1.0):
    path = Path(path)
    if path.exists():
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "target_score_1point"])
        writer.writeheader()
        writer.writerow({"subject_id": subject_id, "target_score_1point": float(score)})


def read_target_score(path, subject_id):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row["subject_id"]) == str(subject_id):
                return float(row["target_score_1point"])
    raise ValueError(f"Subject {subject_id} not found in {path}")


def ensure_channel_first(x):
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D raw array, got shape={x.shape}")
    if x.shape[0] <= 128 and x.shape[1] > x.shape[0]:
        return x
    if x.shape[1] <= 128 and x.shape[0] > x.shape[1]:
        return x.T
    return x


def find_single_file(data_dir, suffix):
    files = sorted(Path(data_dir).glob(f"*{suffix}"))
    if not files:
        raise FileNotFoundError(f"No *{suffix} file found under {data_dir}")
    return files[0]


def load_paths(cfg):
    data_dir = Path(cfg.data_dir)
    return (
        find_single_file(data_dir, "_data.npy"),
        find_single_file(data_dir, "_labels.json"),
        find_single_file(data_dir, "_meta.json"),
    )


def bandpass(data, sfreq, low, high, order=4):
    nyq = sfreq / 2.0
    low_norm = float(low) / nyq
    high_norm = min(float(high) / nyq, 0.999)
    if low_norm <= 0 or high_norm <= low_norm:
        raise ValueError(f"Invalid bandpass range: {low}-{high} Hz")
    sos = butter(order, [low_norm, high_norm], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, data, axis=-1)
    except ValueError:
        return sosfilt(sos, data, axis=-1)


def modality_bandpass_filter(data, cfg):
    out = np.zeros_like(data, dtype=np.float64)
    out[list(cfg.eeg_idx)] = bandpass(data[list(cfg.eeg_idx)], cfg.sfreq, cfg.eeg_band[0], cfg.eeg_band[1], cfg.filter_order)
    out[list(cfg.emg_idx)] = bandpass(data[list(cfg.emg_idx)], cfg.sfreq, cfg.emg_band[0], cfg.emg_band[1], cfg.filter_order)
    out[list(cfg.ecg_idx)] = bandpass(data[list(cfg.ecg_idx)], cfg.sfreq, cfg.ecg_band[0], cfg.ecg_band[1], cfg.filter_order)
    return out


def load_data_array(path, cfg):
    data = ensure_channel_first(np.load(path))
    if data.shape[0] < cfg.n_channels:
        raise ValueError(f"{path} has {data.shape[0]} channels, expected at least {cfg.n_channels}")
    data = data[: cfg.n_channels].astype(np.float64)
    if cfg.data_unit.lower() == "v":
        data = data * 1e6
    return modality_bandpass_filter(data, cfg)


def parse_epoch_id(name):
    head = str(name).split("_", 1)[0]
    if head.startswith("E"):
        try:
            return int(head[1:])
        except ValueError:
            return -1
    return -1


def load_events(labels_path, cfg):
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    events = []
    for row in labels:
        code = int(row.get("code", -1))
        if code not in (cfg.imagined_code, cfg.warm_code):
            continue
        events.append(
            {
                "epoch": parse_epoch_id(row.get("name", "")),
                "timestamp_sec": float(row["timestamp_sec"]),
                "code": code,
                "name": row.get("name", ""),
            }
        )
    return events


def extract_raw_window(raw, start_sec, cfg):
    start = int(round(float(start_sec) * cfg.sfreq))
    length = int(round(float(cfg.window_sec) * cfg.sfreq))
    end = start + length
    if start < 0 or end > raw.shape[1]:
        raise ValueError(f"Window [{start}:{end}] is outside raw length {raw.shape[1]}")
    return raw[: cfg.n_channels, start:end].astype(np.float32, copy=False)


class RawWindowStandardizer:
    def __init__(self, eps=1e-6, clip_value=8.0):
        self.eps = float(eps)
        self.clip_value = float(clip_value)
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(axis=(0, 1, 3), keepdims=True)
        self.std = x.std(axis=(0, 1, 3), keepdims=True) + self.eps
        return self

    def transform(self, x):
        x = (x - self.mean) / self.std
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, -self.clip_value, self.clip_value)
        return x.astype(np.float32)


def build_warm_prior_windows(cfg):
    npy_path, labels_path, _ = load_paths(cfg)
    raw = load_data_array(npy_path, cfg)
    events = load_events(labels_path, cfg)
    imagined_by_epoch = {e["epoch"]: e for e in events if e["code"] == cfg.imagined_code}
    warm_by_epoch = {e["epoch"]: e for e in events if e["code"] == cfg.warm_code}
    epochs = sorted(set(imagined_by_epoch).intersection(warm_by_epoch))[: int(cfg.n_tasks)]
    if not epochs:
        raise ValueError("No matched imagined/warm epoch pairs were found.")

    warm_windows, prior_windows, rows = [], [], []
    for task_index, epoch in enumerate(epochs):
        imagined_event = imagined_by_epoch[epoch]
        warm_event = warm_by_epoch[epoch]
        prior_windows.append(extract_raw_window(raw, imagined_event["timestamp_sec"], cfg))
        warm_windows.append(extract_raw_window(raw, warm_event["timestamp_sec"], cfg))
        rows.append({"task_index": task_index, "epoch": epoch, "warm_name": warm_event["name"], "prior_name": imagined_event["name"]})

    return (
        np.stack(warm_windows, axis=0).astype(np.float32)[None, ...],
        np.stack(prior_windows, axis=0).astype(np.float32)[None, ...],
        rows,
    )


def modality_frequency_grid(modality, cfg):
    if modality == "eeg":
        return np.linspace(cfg.eeg_band[0], cfg.eeg_band[1], cfg.n_freqs, dtype=np.float64)
    if modality == "emg":
        return np.linspace(cfg.emg_band[0], cfg.emg_band[1], cfg.n_freqs, dtype=np.float64)
    if modality == "ecg":
        return np.linspace(cfg.ecg_band[0], cfg.ecg_band[1], cfg.n_freqs, dtype=np.float64)
    raise ValueError(f"Unknown modality: {modality}")


def morlet_wavelet(freq_hz, sfreq, n_cycles, max_len):
    sigma_t = n_cycles / (2.0 * np.pi * max(float(freq_hz), 1e-6))
    half_samples = int(np.ceil(3.0 * sigma_t * sfreq))
    half_samples = max(4, min(half_samples, max_len - 1))
    t = np.arange(-half_samples, half_samples + 1, dtype=np.float64) / sfreq
    wavelet = np.exp(2j * np.pi * float(freq_hz) * t) * np.exp(-(t ** 2) / (2.0 * sigma_t ** 2))
    wavelet = wavelet - wavelet.mean()
    norm = np.sqrt(np.sum(np.abs(wavelet) ** 2)) + 1e-12
    return wavelet / norm


def cwt_morlet_channel(x, freqs, cfg):
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.empty((len(freqs), len(x)), dtype=np.float32)
    for i, freq in enumerate(freqs):
        wavelet = morlet_wavelet(float(freq), cfg.sfreq, cfg.morlet_cycles, max_len=len(x))
        conv = fftconvolve(x, wavelet.conj()[::-1], mode="same")
        out[i] = np.abs(conv).astype(np.float32)
    return out


def compute_modality_tfr(window, modality, cfg):
    freqs = modality_frequency_grid(modality, cfg)
    tfr = np.empty((window.shape[0], cfg.n_freqs, window.shape[1]), dtype=np.float32)
    for ch in range(window.shape[0]):
        tfr[ch] = cwt_morlet_channel(window[ch], freqs, cfg)
    if cfg.tf_log_amplitude:
        tfr = np.log1p(tfr)
    if cfg.tf_per_channel_standardize:
        mean = tfr.mean(axis=(1, 2), keepdims=True)
        std = tfr.std(axis=(1, 2), keepdims=True) + 1e-6
        tfr = (tfr - mean) / std
        tfr = np.clip(np.nan_to_num(tfr, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
    return tfr.astype(np.float32)


def segment_to_time_frequency(segment, cfg):
    tf_cft = np.empty((cfg.n_channels, cfg.n_freqs, cfg.n_times), dtype=np.float32)
    tf_cft[list(cfg.eeg_idx)] = compute_modality_tfr(segment[list(cfg.eeg_idx)], "eeg", cfg)
    tf_cft[list(cfg.emg_idx)] = compute_modality_tfr(segment[list(cfg.emg_idx)], "emg", cfg)
    tf_cft[list(cfg.ecg_idx)] = compute_modality_tfr(segment[list(cfg.ecg_idx)], "ecg", cfg)
    return np.transpose(tf_cft, (1, 0, 2)).astype(np.float32)


def load_or_compute_tfr_pair(warm_raw, prior_raw, cfg, rows):
    cache_dir = Path(cfg.output_dir) / "morlet_tfr_cache"
    if cfg.cache_time_frequency:
        cache_dir.mkdir(parents=True, exist_ok=True)
    warm_tfr, prior_tfr = [], []
    for k in range(warm_raw.shape[1]):
        cache_path = cache_dir / f"epoch_{rows[k]['epoch']:02d}_warm_prior_tfr.npz"
        if cfg.cache_time_frequency and cache_path.exists():
            obj = np.load(cache_path)
            warm = obj["warm"].astype(np.float32)
            prior = obj["prior"].astype(np.float32)
        else:
            warm = segment_to_time_frequency(warm_raw[0, k], cfg)
            prior = segment_to_time_frequency(prior_raw[0, k], cfg)
            if cfg.cache_time_frequency:
                np.savez_compressed(cache_path, warm=warm.astype(np.float16), prior=prior.astype(np.float16))
        warm_tfr.append(warm)
        prior_tfr.append(prior)
    return (
        np.stack(warm_tfr, axis=0)[None, ...].astype(np.float32),
        np.stack(prior_tfr, axis=0)[None, ...].astype(np.float32),
    )


def to_model_config(cfg):
    return ModelConfig(
        n_freqs=cfg.n_freqs,
        n_times=cfg.n_times,
        raw_hidden_dim=cfg.raw_hidden_dim,
        task_embed_dim=cfg.task_embed_dim,
        task_att_hidden=cfg.task_att_hidden,
        dropout=cfg.dropout,
        task_mean_residual_ratio=cfg.task_mean_residual_ratio,
        prior_lambda=cfg.prior_lambda,
    )


def write_result_csv(path, subject_id, target_score, fitted_score, final_loss, task_weight):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject_id", "target_score_1point", "fitted_score_1point", "fit_loss", "task_attention"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "subject_id": subject_id,
                "target_score_1point": f"{target_score:.6f}",
                "fitted_score_1point": f"{fitted_score:.6f}",
                "fit_loss": f"{final_loss:.8f}",
                "task_attention": ";".join(f"{float(v):.6f}" for v in task_weight),
            }
        )


def run_experiment():
    cfg = ExperimentConfig()
    cfg.n_times = int(cfg.window_sec * cfg.sfreq)
    set_seed(cfg.random_state)
    safe_mkdir(cfg.output_dir)
    ensure_target_csv(cfg.target_csv, cfg.subject_id, score=1.0)

    target_score = read_target_score(cfg.target_csv, cfg.subject_id)
    warm_raw, prior_raw, rows = build_warm_prior_windows(cfg)
    standardizer = RawWindowStandardizer().fit(np.concatenate([warm_raw, prior_raw], axis=1))
    warm_raw = standardizer.transform(warm_raw)
    prior_raw = standardizer.transform(prior_raw)
    warm_tf_np, prior_tf_np = load_or_compute_tfr_pair(warm_raw, prior_raw, cfg, rows)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    warm_tf = torch.from_numpy(warm_tf_np).float().to(device)
    prior_tf = torch.from_numpy(prior_tf_np).float().to(device)
    target = torch.tensor([target_score], dtype=torch.float32, device=device)

    model = build_model(to_model_config(cfg)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_score = 0.0
    best_weight = np.ones(cfg.n_tasks, dtype=np.float32) / float(cfg.n_tasks)

    for _ in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred, aux = model(warm_tf, prior_tf)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_score = float(pred.detach().cpu().item())
            best_weight = aux["task_weight"].detach().cpu().numpy()[0]

    result_path = Path(cfg.output_dir) / "fit_score.csv"
    write_result_csv(result_path, cfg.subject_id, target_score, best_score, best_loss, best_weight)

    print(f"Saved: {result_path}")
    print(f"Target={target_score:.6f}, fitted={best_score:.6f}")


if __name__ == "__main__":
    run_experiment()
