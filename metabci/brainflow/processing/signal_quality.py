# -*- coding: utf-8 -*-
"""
Real-time signal quality monitor — per-channel contact/noise assessment.

Provides ``SignalQualityMonitor``, a lightweight class that tracks
electrode contact quality and line-noise contamination for each channel
using EMA-decayed statistics.  Designed for online use inside the
display loop; no GUI dependencies.

Logs quality events via ``metabci.brainflow.logger`` for integration
with the rest of the framework.  The line-noise measurement methodology
complements ``metabci.brainda.algorithms.feature_analysis.FrequencyAnalysis
.signal_noise_ratio()``, which provides FFT-based SNR for offline analysis.

Examples
--------
>>> sqm = SignalQualityMonitor(n_channels=8, fs=250.0, line_freq=50.0)
>>> quality = sqm.update(filtered_data)  # filtered_data: (n_ch, n_new)
>>> for ch_name, status in zip(names, sqm.get_channel_status()):
...     print(ch_name, status)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import butter, sosfilt

try:
    from ..logger import get_logger
except ImportError:
    from metabci.brainflow.logger import get_logger  # type: ignore[no-redef]

logger = get_logger("signal_quality")


class SignalQualityMonitor:
    """Per-channel signal quality tracking with EMA statistics.

    Tracks running variance, line-noise power ratio, and flat-signal
    detection.  Returns human-readable channel statuses suitable for
    driving a GUI quality indicator bar.

    The line-noise measurement uses a narrow-band SOS filter (same
    ``scipy.signal.butter`` / ``sosfilt`` toolchain as brainda's
    ``FrequencyAnalysis``).  For offline FFT-based SNR analysis, see
    ``metabci.brainda.algorithms.feature_analysis.FrequencyAnalysis
    .signal_noise_ratio()``.

    Parameters
    ----------
    n_channels : int
        Number of data channels.
    fs : float
        Sampling rate in Hz.
    line_freq : float
        Power-line frequency (50.0 or 60.0 Hz).
    channel_names : list of str, optional
        Display names for each channel.
    variance_tau : float
        EMA time constant (seconds) for running variance.
    flat_threshold_ratio : float
        A channel is flagged as *flat* when its running variance drops
        below this fraction of the median variance across all channels.
    flat_persistence : float
        Seconds a channel must stay below the threshold before being
        reported as flat.
    """

    def __init__(
        self,
        n_channels: int,
        fs: float,
        line_freq: float = 50.0,
        channel_names: Optional[List[str]] = None,
        variance_tau: float = 2.0,
        flat_threshold_ratio: float = 0.05,
        flat_persistence: float = 1.5,
        high_var_ratio: float = 4.0,
        high_var_persistence: float = 0.8,
    ) -> None:
        self._n_ch = int(n_channels)
        self._fs = float(fs)
        self._line_freq = float(line_freq)
        self._variance_tau = float(variance_tau)
        self._flat_ratio = float(flat_threshold_ratio)
        self._flat_secs = float(flat_persistence)
        self._high_var_ratio = float(high_var_ratio)
        self._high_var_secs = float(high_var_persistence)

        self._names = (
            list(channel_names)
            if channel_names is not None
            else [f"Ch{i + 1}" for i in range(self._n_ch)]
        )

        # EMA decay factors
        self._alpha_var = 1.0 / max(1.0, fs * variance_tau)
        self._alpha_lnr = 1.0 / max(1.0, fs * 5.0)

        # Running state
        self._running_var = np.ones(self._n_ch, dtype=np.float64)
        self._running_lnr = np.zeros(self._n_ch, dtype=np.float64)
        self._flat_counter = np.zeros(self._n_ch, dtype=np.float64)
        self._high_var_counter = np.zeros(self._n_ch, dtype=np.float64)
        self._samples_seen: int = 0
        self._warmup_samples: int = int(fs * 3.0)

        # Narrow-band bandpass SOS for line-noise measurement
        self._ln_sos = self._design_line_noise_filter(line_freq, fs)

        logger.info(
            "SignalQualityMonitor initialised: n_ch=%d, fs=%.1f Hz, "
            "line_freq=%.1f Hz, variance_tau=%.1f s",
            n_channels, fs, line_freq, variance_tau,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Ingest a chunk of *filtered* data and update quality metrics.

        Parameters
        ----------
        data : np.ndarray
            Shape ``(n_channels, n_new)`` — bandpass-filtered data in uV.

        Returns
        -------
        dict
            Keys: ``"variance"``, ``"line_noise_ratio"``, ``"is_flat"``,
            ``"is_noisy"``, ``"is_high_var"``, ``"contact_quality"`` (0–100).
        """
        if data.size == 0 or data.shape[1] == 0:
            return self._empty_result()

        n_new = data.shape[1]
        self._samples_seen += n_new

        for i in range(n_new):
            chunk = data[:, i].astype(np.float64)
            sq = chunk * chunk
            self._running_var = (
                self._alpha_var * sq
                + (1.0 - self._alpha_var) * self._running_var
            )

        if n_new > 1 and self._ln_sos is not None:
            ln_power = self._estimate_line_noise_power(data)
            total_power = np.var(data, axis=1)
            safe_total = np.maximum(total_power, 1e-12)
            ln_ratio = np.clip(ln_power / safe_total, 0.0, 1.0)
            self._running_lnr = (
                self._alpha_lnr * ln_ratio
                + (1.0 - self._alpha_lnr) * self._running_lnr
            )

        # Flat detection
        median_var = float(np.median(self._running_var))
        flat_threshold = self._flat_ratio * max(median_var, 1e-12)
        for ch in range(self._n_ch):
            if self._running_var[ch] < flat_threshold:
                self._flat_counter[ch] += n_new
            else:
                self._flat_counter[ch] = 0
        is_flat = self._flat_counter >= (self._flat_secs * self._fs)

        # High-variance detection
        ref_var = (
            float(np.percentile(self._running_var, 25))
            if self._n_ch >= 3
            else median_var
        )
        ref_var = max(ref_var, 1e-12)
        high_var_threshold = self._high_var_ratio * ref_var
        for ch in range(self._n_ch):
            if self._running_var[ch] > high_var_threshold:
                self._high_var_counter[ch] += n_new
            else:
                self._high_var_counter[ch] = 0
        is_high_var = self._high_var_counter >= (self._high_var_secs * self._fs)

        # Noisy detection
        is_noisy = self._running_lnr > 0.20

        # Contact quality score 0–100
        quality = np.full(self._n_ch, 100.0, dtype=np.float64)
        quality = np.where(is_high_var, 15.0, quality)
        quality = np.where(is_flat, 20.0, quality)
        quality = np.where(is_noisy & ~is_flat & ~is_high_var, 45.0, quality)
        rel_var = self._running_var / max(median_var, 1e-12)
        low_var_penalty = np.clip((1.0 - rel_var) * 50.0, 0, 30)
        high_var_penalty = np.clip((rel_var - 3.0) * 5.0, 0, 25)
        quality = np.where(
            (~is_flat) & (~is_noisy) & (~is_high_var),
            np.clip(100.0 - low_var_penalty - high_var_penalty, 60.0, 100.0),
            quality,
        )

        return {
            "variance": self._running_var.copy(),
            "line_noise_ratio": self._running_lnr.copy(),
            "is_flat": is_flat.copy(),
            "is_noisy": is_noisy.copy(),
            "is_high_var": is_high_var.copy(),
            "contact_quality": quality,
        }

    def get_channel_status(self) -> List[str]:
        """Return one of ``"good"``, ``"poor_contact"``, ``"noisy"``,
        ``"flat"``, ``"unknown"`` per channel.
        """
        if self._samples_seen < self._warmup_samples:
            return ["unknown"] * self._n_ch

        statuses: List[str] = []
        flat_samples = self._flat_secs * self._fs
        hv_samples = self._high_var_secs * self._fs
        for ch in range(self._n_ch):
            if self._flat_counter[ch] >= flat_samples:
                statuses.append("flat")
            elif self._high_var_counter[ch] >= hv_samples:
                statuses.append("poor_contact")
            elif self._running_lnr[ch] > 0.20:
                statuses.append("noisy")
            else:
                statuses.append("good")

        bad_count = sum(1 for s in statuses if s != "good")
        if bad_count > 0:
            logger.debug(
                "Signal quality: %d/%d channels good, issues: %s",
                len(statuses) - bad_count, len(statuses),
                ", ".join(
                    f"{self._names[i]}={statuses[i]}"
                    for i, s in enumerate(statuses) if s != "good"
                ),
            )
        return statuses

    def set_line_freq(self, freq: float) -> None:
        """Change the target line frequency and rebuild the measurement filter."""
        self._line_freq = float(freq)
        self._ln_sos = self._design_line_noise_filter(freq, self._fs)
        logger.info("Line frequency changed to %.1f Hz", freq)

    def reset(self) -> None:
        """Clear all running statistics."""
        self._running_var = np.ones(self._n_ch, dtype=np.float64)
        self._running_lnr = np.zeros(self._n_ch, dtype=np.float64)
        self._flat_counter = np.zeros(self._n_ch, dtype=np.float64)
        self._high_var_counter = np.zeros(self._n_ch, dtype=np.float64)
        self._samples_seen = 0
        logger.debug("SignalQualityMonitor reset")

    @property
    def channel_names(self) -> List[str]:
        return list(self._names)

    # ------------------------------------------------------------------
    # Internal helpers — same scipy toolchain as brainda FrequencyAnalysis
    # ------------------------------------------------------------------

    def _design_line_noise_filter(
        self, f0: float, fs: float
    ) -> Optional[np.ndarray]:
        """Narrow-band bandpass around *f0* for line-noise power measurement.

        Uses the same ``scipy.signal.butter`` / ``sosfilt`` pipeline as
        ``metabci.brainda.algorithms.feature_analysis.FrequencyAnalysis``
        for filter design consistency.
        """
        nyq = 0.5 * fs
        if f0 >= nyq * 0.95:
            logger.warning(
                "Line frequency %.1f Hz too close to Nyquist (%.1f Hz), "
                "disabling line-noise monitor", f0, nyq,
            )
            return None
        low = max(1e-4, (f0 - 1.0) / nyq)
        high = min(0.99, (f0 + 1.0) / nyq)
        return butter(4, [low, high], btype="band", output="sos")

    def _estimate_line_noise_power(self, data: np.ndarray) -> np.ndarray:
        """Apply the narrow-band line-noise filter and return per-channel
        RMS power of the filtered output."""
        if self._ln_sos is None:
            return np.zeros(data.shape[0], dtype=np.float64)
        filtered = sosfilt(self._ln_sos, data, axis=1)
        return np.mean(filtered * filtered, axis=1)

    def _empty_result(self) -> Dict[str, np.ndarray]:
        return {
            "variance": self._running_var.copy(),
            "line_noise_ratio": self._running_lnr.copy(),
            "is_flat": np.zeros(self._n_ch, dtype=bool),
            "is_noisy": np.zeros(self._n_ch, dtype=bool),
            "is_high_var": np.zeros(self._n_ch, dtype=bool),
            "contact_quality": np.full(self._n_ch, 100.0, dtype=np.float64),
        }
