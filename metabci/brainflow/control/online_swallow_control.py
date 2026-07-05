# -*- coding: utf-8 -*-
"""Online swallow-imagery inference used by the control GUI.

The offline competition classifier is kept in
``metabci.brainflow.competition_algorithms.classification_inference``.  This
module wraps the same preprocessing/model path for a single live 5 s window so
the GUI can decide whether to trigger the ESP32B controller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


class OnlineSwallowIntentDetector:
    """Run the part2 swallow-imagery classifier on one live window."""

    def __init__(
        self,
        project_root: str | Path,
        model_path: str | Path | None = None,
        srate: float = 500.0,
        threshold: Optional[float] = None,
        device: str = "cpu",
    ):
        self.project_root = Path(project_root)
        self.model_path = Path(model_path) if model_path else None
        self.srate = float(srate)
        self.device_name = device
        self.threshold = float(threshold) if threshold is not None else 0.5
        self.mode = "heuristic"
        self.error = ""
        self.model = None
        self.cfg = None
        self.standardizer = None
        self.torch = None
        self.ci = None

        self._try_load_model()

    def _try_load_model(self) -> None:
        try:
            import torch
            from metabci.brainflow.competition_algorithms import (
                find_default_classifier_model,
            )
            from metabci.brainflow.competition_algorithms import (
                classification_inference as ci,
            )

            checkpoint_path = self.model_path or find_default_classifier_model(self.project_root)
            if not checkpoint_path or not Path(checkpoint_path).is_file():
                self.error = "未找到 final_model.pt，使用启发式在线判别。"
                return

            checkpoint = ci.safe_torch_load(Path(checkpoint_path), map_location=self.device_name)
            cfg = ci.cfg_from_checkpoint(checkpoint)
            cfg.sfreq = float(self.srate)
            cfg.cache_tfr_to_disk = False
            cfg.num_workers = 0
            cfg.batch_size = 1

            standardizer = ci.standardizer_from_state(checkpoint["standardizer_state"])
            device = torch.device(self.device_name)
            model = ci.TimeFrequencyRawLocalGlobalNet(cfg).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self.torch = torch
            self.ci = ci
            self.cfg = cfg
            self.standardizer = standardizer
            self.model = model
            self.model_path = Path(checkpoint_path)
            self.threshold = float(checkpoint.get("threshold", self.threshold))
            self.mode = "final_model"
            self.error = ""
        except Exception as exc:
            self.mode = "heuristic"
            self.error = f"模型加载失败，使用启发式在线判别：{exc}"
            self.model = None

    @property
    def window_samples(self) -> int:
        if self.cfg is not None:
            return int(round(float(self.cfg.sfreq) * float(self.cfg.window_sec)))
        return int(round(self.srate * 5.0))

    def predict(self, window: np.ndarray) -> dict[str, Any]:
        """Return detection result for ``window`` shaped ``(channels, samples)``."""
        data = self._prepare_window(window)
        if self.model is not None:
            return self._predict_with_model(data)
        return self._predict_with_heuristic(data)

    def _prepare_window(self, window: np.ndarray) -> np.ndarray:
        data = np.asarray(window, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"实时窗口必须是二维数组，当前 shape={data.shape}")
        if data.shape[0] > data.shape[1] and data.shape[1] <= 128:
            data = data.T

        target_channels = int(getattr(self.cfg, "n_channels", 16) if self.cfg else 16)
        if data.shape[0] < target_channels:
            pad = np.zeros((target_channels - data.shape[0], data.shape[1]), dtype=data.dtype)
            data = np.vstack([data, pad])
        elif data.shape[0] > target_channels:
            data = data[:target_channels]

        target_samples = self.window_samples
        if data.shape[1] < target_samples:
            pad = np.zeros((data.shape[0], target_samples - data.shape[1]), dtype=data.dtype)
            data = np.hstack([pad, data])
        elif data.shape[1] > target_samples:
            data = data[:, -target_samples:]

        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    def _predict_with_model(self, data: np.ndarray) -> dict[str, Any]:
        assert self.model is not None
        assert self.cfg is not None
        assert self.standardizer is not None
        assert self.torch is not None
        assert self.ci is not None

        filtered = self.ci.modality_bandpass_filter(data, self.cfg)
        x = self.standardizer.transform(filtered[np.newaxis, ...])
        dataset = self.ci.TFRWindowDataset(
            x,
            np.asarray([0], dtype=np.float32),
            self.cfg,
            cache_dir=None,
            split_name="online",
        )
        eeg, emg, ecg, _ = dataset[0]
        device = self.torch.device(self.device_name)
        with self.torch.no_grad():
            logits, aux = self.model(
                eeg.unsqueeze(0).to(device),
                emg.unsqueeze(0).to(device),
                ecg.unsqueeze(0).to(device),
                return_aux=True,
            )
            prob = float(self.torch.sigmoid(logits).detach().cpu().numpy()[0])

        return {
            "detected": bool(prob >= self.threshold),
            "confidence": prob,
            "threshold": self.threshold,
            "mode": self.mode,
            "label": "想象吞咽" if prob >= self.threshold else "非吞咽意图",
            "model_path": str(self.model_path) if self.model_path else "",
            "aux": {
                "modality_weight": aux["modality_weight"].detach().cpu().numpy().tolist(),
            },
        }

    def _predict_with_heuristic(self, data: np.ndarray) -> dict[str, Any]:
        emg_idx = [6, 7, 8, 9, 14, 15]
        eeg_idx = [0, 1, 2, 3, 4, 5, 10, 11, 13]
        emg = data[[i for i in emg_idx if i < data.shape[0]]]
        eeg = data[[i for i in eeg_idx if i < data.shape[0]]]
        emg_rms = float(np.sqrt(np.mean(np.square(emg)))) if emg.size else 0.0
        eeg_rms = float(np.sqrt(np.mean(np.square(eeg)))) if eeg.size else 0.0
        score = 0.65 * np.tanh(emg_rms / 35.0) + 0.35 * np.tanh(eeg_rms / 18.0)
        confidence = float(np.clip(score, 0.0, 1.0))
        return {
            "detected": bool(confidence >= self.threshold),
            "confidence": confidence,
            "threshold": self.threshold,
            "mode": self.mode,
            "label": "想象吞咽" if confidence >= self.threshold else "非吞咽意图",
            "model_path": "",
            "warning": self.error,
            "features": {
                "emg_rms": emg_rms,
                "eeg_rms": eeg_rms,
            },
        }
