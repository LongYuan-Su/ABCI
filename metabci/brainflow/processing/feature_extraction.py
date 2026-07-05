# -*- coding: utf-8 -*-
"""
Real-time feature extraction — band power, Hjorth parameters, and time-domain
metrics.  Designed for online use inside the display loop of ``real_time_eeg.py``;
no GUI or worker dependencies.

Leverages ``metabci.brainda.algorithms.feature_analysis.FrequencyAnalysis``
for band definitions and PSD methodology; the online ring-buffer architecture
complements the offline trial-based analysis in brainda.

Examples
--------
>>> extractor = FeatureExtractor(fs=250.0, n_channels=8)
>>> extractor.feed(filtered_data)   # accumulates into ring buffer
>>> feats = extractor.latest        # None until first extraction window fills
>>> if feats:
...     print(feats["band_alpha"])  # shape (8,)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from ..logger import get_logger
except ImportError:
    from metabci.brainflow.logger import get_logger  # type: ignore[no-redef]

logger = get_logger("feature_extraction")


class FeatureExtractor:
    """Online band-power and time-domain feature extraction.

    Accumulates data into an internal ring buffer and computes features
    periodically (every *step_seconds*).  All operations are vectorised
    over channels for speed.

    The band definitions align with
    ``metabci.brainda.algorithms.feature_analysis.FrequencyAnalysis``
    conventions so that offline-trained models and online features
    share the same frequency ranges.

    Parameters
    ----------
    fs : float
        Sampling rate in Hz.
    n_channels : int
        Number of data channels.
    window_seconds : float
        FFT / analysis window duration in seconds.
    step_seconds : float
        How often to recompute features.
    bands : dict, optional
        Mapping ``{name: (low_Hz, high_Hz)}``.  If *None*, the default
        clinical bands are used (compatible with brainda FrequencyAnalysis).
    time_features : bool
        Whether to compute Hjorth, zero-crossing, and line-length features.
    channel_names : list of str, optional
        Labels for each channel.
    """

    # Default clinical frequency bands — align with brainda convention
    # See: metabci.brainda.algorithms.feature_analysis.FrequencyAnalysis
    FREQ_BANDS: Dict[str, Tuple[float, float]] = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    def __init__(
        self,
        fs: float,
        n_channels: int,
        window_seconds: float = 2.0,
        step_seconds: float = 0.5,
        bands: Optional[Dict[str, Tuple[float, float]]] = None,
        time_features: bool = True,
        channel_names: Optional[List[str]] = None,
    ) -> None:
        self._fs = float(fs)
        self._n_ch = int(n_channels)
        self._window_sec = float(window_seconds)
        self._step_sec = float(step_seconds)

        self._window_samples = int(round(fs * window_seconds))
        self._step_samples = int(round(fs * step_seconds))

        self._bands = bands if bands is not None else dict(self.FREQ_BANDS)
        self._band_names = list(self._bands.keys())
        self._do_time = bool(time_features)

        if channel_names is not None:
            self._names = list(channel_names)
        else:
            self._names = [f"Ch{i + 1}" for i in range(self._n_ch)]

        # -- Ring buffer ------------------------------------------------------
        self._buffer: np.ndarray = np.zeros(
            (self._n_ch, self._window_samples), dtype=np.float64)
        self._write_pos: int = 0        # next write position
        self._samples_since_extract: int = 0

        # -- Latest features cache --------------------------------------------
        self._latest: Optional[Dict[str, np.ndarray]] = None
        self._n_extractions: int = 0

        # -- Pre-compute FFT frequency axis and band bin masks ----------------
        self._freqs = np.fft.rfftfreq(self._window_samples, d=1.0 / fs)
        self._band_masks: Dict[str, np.ndarray] = {}
        for band_name, (low, high) in self._bands.items():
            mask = (self._freqs >= low) & (self._freqs < high)
            self._band_masks[band_name] = mask

        # -- Hamming window for FFT -------------------------------------------
        self._hamming = np.hamming(self._window_samples)

        logger.info(
            "FeatureExtractor initialised: fs=%.1f Hz, n_ch=%d, "
            "window=%.1f s, step=%.1f s, bands=%s",
            fs, n_channels, window_seconds, step_seconds,
            self._band_names,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, data: np.ndarray) -> bool:
        """Accumulate *data* into the analysis buffer.

        Parameters
        ----------
        data : np.ndarray
            Shape ``(n_channels, n_new)`` — filtered data in uV.

        Returns
        -------
        bool
            ``True`` if new features were just computed this call.
        """
        if data.size == 0:
            return False

        n_new = data.shape[1]
        ch = min(data.shape[0], self._n_ch)

        # Write into ring buffer
        buf_len = self._buffer.shape[1]
        for i in range(n_new):
            pos = self._write_pos % buf_len
            self._buffer[:ch, pos] = data[:ch, i]
            self._write_pos += 1

        self._samples_since_extract += n_new
        if self._samples_since_extract >= self._step_samples:
            self._samples_since_extract = 0
            self._extract_features()
            return True
        return False

    @property
    def latest(self) -> Optional[Dict[str, np.ndarray]]:
        """Most recent feature dict, or *None* if no extraction has run yet.

        Dict keys (numpy arrays of shape ``(n_channels,)``):

        **Band power** (absolute):
          ``band_delta``, ``band_theta``, ``band_alpha``,
          ``band_beta``, ``band_gamma``

        **Band power** (relative — ``_rel`` suffix):
          ``band_delta_rel``, ``band_theta_rel``, ``band_alpha_rel``,
          ``band_beta_rel``, ``band_gamma_rel``

        **Time-domain** (if *time_features* is ``True``):
          ``hjorth_activity``, ``hjorth_mobility``,
          ``hjorth_complexity``, ``zero_crossing_rate``,
          ``line_length``
        """
        return self._latest

    def get_band_power(self, band: str, relative: bool = False) -> np.ndarray:
        """Convenience accessor for a single band.

        Parameters
        ----------
        band : str
            One of ``"delta"``, ``"theta"``, ``"alpha"``, ``"beta"``, ``"gamma"``.
        relative : bool
            Return relative power instead of absolute.

        Returns
        -------
        np.ndarray
            Shape ``(n_channels,)``, or zeros if no features are available.
        """
        if self._latest is None:
            return np.zeros(self._n_ch, dtype=np.float64)
        key = f"band_{band}_rel" if relative else f"band_{band}"
        return self._latest.get(key, np.zeros(self._n_ch, dtype=np.float64))

    def reset(self) -> None:
        """Clear the ring buffer and cached features."""
        self._buffer.fill(0.0)
        self._write_pos = 0
        self._samples_since_extract = 0
        self._latest = None
        self._n_extractions = 0

    @property
    def band_names(self) -> List[str]:
        return list(self._band_names)

    @property
    def channel_names(self) -> List[str]:
        return list(self._names)

    @property
    def window_seconds(self) -> float:
        return self._window_sec

    @property
    def step_seconds(self) -> float:
        return self._step_sec

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _extract_features(self) -> None:
        """Compute all features over the current ring buffer content."""
        # Total samples available
        available = min(self._write_pos, self._window_samples)
        if available < 4:  # need at least a few samples
            return

        # Get current window (wrapped if needed)
        buf_len = self._buffer.shape[1]
        if self._write_pos <= buf_len:
            # First fill — use [0 : write_pos]; pad with zeros if not full
            window = np.zeros((self._n_ch, self._window_samples),
                              dtype=np.float64)
            window[:, :available] = self._buffer[:, :available]
        else:
            wp = self._write_pos % buf_len
            window = np.concatenate(
                [self._buffer[:, wp:], self._buffer[:, :wp]], axis=1)

        feats: Dict[str, np.ndarray] = {}

        # -- Band power via FFT (methodology aligned with
        #    brainda.algorithms.feature_analysis.FrequencyAnalysis
        #    .power_spectrum_periodogram / .signal_noise_ratio) --------------
        feats.update(self._compute_band_power(window))

        # -- Time-domain features ---------------------------------------------
        if self._do_time:
            feats.update(self._compute_time_features(window))

        self._latest = feats
        self._n_extractions += 1

    def _compute_band_power(
        self, window: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """FFT-based band power (absolute and relative).

        Uses the same FFT methodology as
        ``metabci.brainda.algorithms.feature_analysis.FrequencyAnalysis``
        (via ``scipy.signal.periodogram`` / ``scipy.fftpack.fft``) but
        implemented in a vectorised online fashion across all channels.
        """
        n_ch, n_samp = window.shape
        # Apply Hamming window
        windowed = window * self._hamming[:n_samp]
        # rFFT per channel (vectorised)
        fft = np.fft.rfft(windowed, axis=1)    # (n_ch, n_freqs)
        power = np.abs(fft) ** 2               # squared magnitude

        features: Dict[str, np.ndarray] = {}
        total_power = np.sum(power, axis=1)     # per-channel total
        safe_total = np.maximum(total_power, 1e-12)

        for band_name in self._band_names:
            mask = self._band_masks.get(band_name)
            if mask is None or not np.any(mask):
                features[f"band_{band_name}"] = np.zeros(n_ch,
                                                          dtype=np.float64)
                features[f"band_{band_name}_rel"] = np.zeros(n_ch,
                                                              dtype=np.float64)
                continue
            # Ensure mask length matches power shape (handles edge cases)
            mask_trimmed = mask[:power.shape[1]]
            band_power = np.sum(power[:, mask_trimmed], axis=1)
            features[f"band_{band_name}"] = band_power
            # Relative power: fraction of total (0–1)
            features[f"band_{band_name}_rel"] = np.clip(
                band_power / safe_total, 0.0, 1.0)

        return features

    def _compute_time_features(
        self, window: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Hjorth parameters, zero-crossing rate, line-length.

        These complement the ERP features provided by
        ``metabci.brainda.algorithms.feature_analysis.TimeAnalysis``
        (peak_amplitude, average_amplitude, peak_latency, average_latency),
        adding online Hjorth descriptors not available in the offline API.
        """
        n_ch, _n_samp = window.shape
        features: Dict[str, np.ndarray] = {}

        # Hjorth Activity = variance
        activity = np.var(window, axis=1)
        features["hjorth_activity"] = activity

        # Hjorth Mobility = sqrt(var(diff1) / var(signal))
        diff1 = np.diff(window, axis=1)
        var_diff1 = np.var(diff1, axis=1)
        safe_activity = np.maximum(activity, 1e-12)
        mobility = np.sqrt(var_diff1 / safe_activity)
        features["hjorth_mobility"] = mobility

        # Hjorth Complexity = sqrt(var(diff2) / var(diff1)) / mobility
        diff2 = np.diff(diff1, axis=1)
        var_diff2 = np.var(diff2, axis=1)
        safe_var_diff1 = np.maximum(var_diff1, 1e-12)
        safe_mobility = np.maximum(mobility, 1e-12)
        complexity = np.sqrt(var_diff2 / safe_var_diff1) / safe_mobility
        features["hjorth_complexity"] = complexity

        # Zero-crossing rate (normalised to 0–1)
        signs = np.sign(window)
        zero_crossings = np.sum(
            np.abs(np.diff(signs, axis=1)) > 0, axis=1
        ).astype(np.float64)
        features["zero_crossing_rate"] = zero_crossings / max(1, _n_samp - 1)

        # Line length = sum of absolute first differences
        line_len = np.sum(np.abs(diff1), axis=1)
        features["line_length"] = line_len

        return features
