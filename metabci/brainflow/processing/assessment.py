# -*- coding: utf-8 -*-
"""Swallow Assessment Engine — 基于 Fork C 深度学习模型。

使用 ``metabci.brainda.algorithms.deep_learning.swallow_net.SwallowQuantificationNet``
进行端到端吞咽障碍量化评分（0-100），替代原四维度加权评估。

模型输入：三模态 (EEG, EMG, ECG) 1 秒窗口 [B, C, 500]
模型输出：0-100 吞咽障碍风险评分

回退路径（模型不可用时）：基于 CSV 事件计数的保守估算。

Provides: SwallowAssessmentEngine, assess_from_paradigm_log
"""

import csv
import datetime
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from ..logger import get_logger
except ImportError:
    from metabci.brainflow.logger import get_logger  # type: ignore[no-redef]

logger = get_logger("assessment")


class SwallowAssessmentEngine:
    """吞咽功能量化评估引擎 — Fork C SwallowQuantificationNet 端到端评分。

    优先使用 ``SwallowQuantificationNet`` 直接输出 0-100 评分；
    模型不可用时回退到 CSV 事件计数估算。

    Parameters
    ----------
    patient_id : str
    model_path : str or None
        Pre-trained SwallowQuantificationNet checkpoint path.
    device : str
        Torch device (``"cpu"`` or ``"cuda"``).
    """

    def __init__(self, patient_id: str = "",
                 model_path: Optional[str] = None,
                 device: str = "cpu"):
        self.patient_id = patient_id
        self.model_path = model_path
        self.device = device
        self._model = None
        self.dimensions: Dict[str, Any] = {}

    def _load_model(self):
        """Lazy-load SwallowQuantificationNet from checkpoint."""
        if self._model is not None:
            return self._model
        try:
            from metabci.brainda.algorithms.deep_learning.swallow_trainer import (
                load_swallow_quantifier)
            if self.model_path and os.path.isfile(self.model_path):
                self._model = load_swallow_quantifier(self.model_path, self.device)
            else:
                from metabci.brainda.algorithms.deep_learning.swallow_net import (
                    SwallowQuantificationNet)
                self._model = SwallowQuantificationNet().to(self.device)
                self._model.eval()
            logger.info("SwallowQuantificationNet loaded")
        except Exception as e:
            logger.warning("SwallowQuantificationNet not available: %s", e)
            self._model = False  # mark as tried-and-failed
        return self._model if self._model is not False else None

    def evaluate(
        self,
        paradigm_log_path: str = "",
        epochs_data: Optional[Dict[str, np.ndarray]] = None,
    ) -> dict:
        """运行评估并返回报告。

        优先：SwallowQuantificationNet → 直接 0-100 评分
        回退：CSV 事件计数 → 估算评分

        Parameters
        ----------
        paradigm_log_path : str
        epochs_data : dict, optional
            ``{"eeg": (n_trials, n_ch, n_times),
               "emg": (n_trials, n_ch, n_times),
               "ecg": (n_trials, n_ch, n_times)}``

        Returns
        -------
        report : dict
        """
        events = []
        if paradigm_log_path and os.path.isfile(paradigm_log_path):
            events = self._parse_log(paradigm_log_path)

        # --- Primary: SwallowQuantificationNet ---
        model_score = None
        if epochs_data and ("eeg" in epochs_data or "emg" in epochs_data):
            model_score = self._score_with_model(epochs_data, events)

        if model_score is not None:
            dimensions = {"swallow_net": model_score}
            composite = model_score["score"]
        else:
            # --- Fallback: event-count estimate ---
            n_imagine = len([e for e in events
                           if "想象吞咽" in e.get("event_name", "")])
            n_water = len([e for e in events
                          if "温水吞咽" in e.get("event_name", "")])
            composite = min(100, 30 + n_imagine * 5 + n_water * 8)
            dimensions = {
                "swallow_net": {
                    "score": composite,
                    "level": self._score_to_level(composite),
                    "note": f"模型不可用，基于 {n_imagine} 想象+{n_water} 温水吞咽事件估算",
                }
            }

        self.dimensions = dimensions
        level = self._score_to_level(composite)
        recs = self._generate_recommendations(composite, dimensions)

        report = {
            "patient_id": self.patient_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "composite_score": composite,
            "composite_level": level,
            "dimensions": dimensions,
            "recommendations": recs,
            "total_events": len(events),
        }

        logger.info("评估完成: patient=%s, score=%d/100 (%s)",
                     self.patient_id, composite, level)
        return report

    # ------------------------------------------------------------------
    # SwallowQuantificationNet scoring
    # ------------------------------------------------------------------

    def _score_with_model(self, epochs_data: dict, events: list) -> dict | None:
        """Run SwallowQuantificationNet on epoch windows → 0-100 score.

        Slices each trial into 1 s windows, runs the model, averages
        per-trial scores, then averages across trials.
        """
        model = self._load_model()
        if model is None or model is False:
            return None

        try:
            import torch

            # Prepare modality tensors
            eeg = epochs_data.get("eeg")  # (n_trials, n_ch, n_times)
            emg = epochs_data.get("emg")
            ecg = epochs_data.get("ecg")

            if eeg is None and emg is None:
                return None

            n_trials = eeg.shape[0] if eeg is not None else emg.shape[0]
            n_times = eeg.shape[2] if eeg is not None else emg.shape[2]

            # Slice into 1 s windows (500 samples at 500 Hz)
            window_len = 500
            stride = 250  # 0.5 s overlap
            trial_scores = []

            for t in range(n_trials):
                windows = []
                for start in range(0, n_times - window_len + 1, stride):
                    end = start + window_len
                    w_eeg = (torch.from_numpy(eeg[t, :, start:end]).float().unsqueeze(0)
                             if eeg is not None else
                             torch.zeros(1, 9, window_len))
                    w_emg = (torch.from_numpy(emg[t, :, start:end]).float().unsqueeze(0)
                             if emg is not None else
                             torch.zeros(1, 6, window_len))
                    w_ecg = (torch.from_numpy(ecg[t, :, start:end]).float().unsqueeze(0)
                             if ecg is not None else
                             torch.zeros(1, 1, window_len))

                    with torch.no_grad():
                        pred = model(w_eeg, w_emg, w_ecg)  # [B]
                    score = float(pred.item())
                    windows.append(np.clip(score, 0.0, 100.0))

                if windows:
                    trial_scores.append(float(np.mean(windows)))

            if not trial_scores:
                return None

            mean_score = round(float(np.mean(trial_scores)))
            mean_score = max(0, min(100, mean_score))

            n_imagine = len([e for e in events
                           if "想象吞咽" in e.get("event_name", "")])

            return {
                "score": mean_score,
                "level": self._score_to_level(mean_score),
                "model": "SwallowQuantificationNet",
                "n_trials": n_trials,
                "n_windows_per_trial": len(windows) if windows else 0,
                "trial_scores": [round(s, 1) for s in trial_scores],
                "note": f"Fork C SwallowQuantificationNet: {n_trials} 试次, "
                        f"{n_imagine} 吞咽事件",
            }
        except Exception as e:
            logger.warning("SwallowQuantificationNet scoring failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_level(score: int) -> str:
        if score >= 85:
            return "优秀"
        elif score >= 70:
            return "良好"
        elif score >= 50:
            return "一般"
        elif score >= 30:
            return "较差"
        else:
            return "严重"

    def _parse_log(self, log_path: str) -> list:
        events: List[Dict] = []
        try:
            with open(log_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    events.append(row)
        except Exception as e:
            logger.warning("日志解析失败: %s", e)
        return events

    def _generate_recommendations(self, composite: float,
                                   dimensions: dict) -> dict:
        threshold = max(0.4, min(0.85, 0.7 - (composite - 50) * 0.005))
        threshold = round(threshold, 2)

        notes = []
        dim = dimensions.get("swallow_net", {})
        score = dim.get("score", composite)

        if score < 50:
            notes.append(
                "吞咽功能评分较低，建议增加训练试次并检查电极接触质量")
        elif score < 70:
            notes.append(
                "吞咽功能中等，建议持续康复训练并定期评估")
        if "model" in dim:
            notes.append(f"评分由 {dim['model']} 端到端模型生成")

        return {
            "confidence_threshold": threshold,
            "decoder_type": "swallow_net",
            "notes": notes,
        }


def assess_from_paradigm_log(
    patient_id: str,
    csv_log_path: str,
    epochs_data: Optional[Dict[str, np.ndarray]] = None,
    model_path: Optional[str] = None,
) -> dict:
    """Parse paradigm log and return SwallowQuantificationNet assessment.

    Parameters
    ----------
    patient_id : str
    csv_log_path : str
    epochs_data : dict, optional
        ``{"eeg": ndarray, "emg": ndarray, "ecg": ndarray}``
    model_path : str, optional
        Path to pre-trained SwallowQuantificationNet checkpoint.

    Returns
    -------
    report : dict
    """
    engine = SwallowAssessmentEngine(
        patient_id=patient_id, model_path=model_path)
    return engine.evaluate(
        paradigm_log_path=csv_log_path, epochs_data=epochs_data)
