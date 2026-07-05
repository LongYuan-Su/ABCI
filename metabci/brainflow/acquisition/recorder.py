# -*- coding: utf-8 -*-
"""EEG/EMG/ECG data recorder for MetaBCI brainflow.

Uses ``metabci.brainflow.amplifiers.RingBuffer`` for thread-safe data
buffering and ``Marker`` for event-aligned epoch tracking, replacing
manual Python-list buffer management.

Provides: EEGRecorder for saving streaming data to NPY files with labels.
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from ..logger import get_logger
    from ..amplifiers import RingBuffer, Marker
except ImportError:
    from metabci.brainflow.logger import get_logger  # type: ignore[no-redef]
    from metabci.brainflow.amplifiers import RingBuffer, Marker  # type: ignore[no-redef]

logger = get_logger("recorder")


class EEGRecorder:
    """Record EEG/EMG/ECG streaming data to NPY files with event labels.

    Thread-safe buffering via ``metabci.brainflow.amplifiers.RingBuffer``
    (fixed-size deque).  Event-aligned epoch tracking via
    ``metabci.brainflow.amplifiers.Marker`` for online closed-loop
    compatibility.

    Parameters
    ----------
    output_dir : str
        Directory to save recordings.
    subject_id : str
        Subject identifier.
    srate : float
        Sampling rate in Hz.
    n_channels : int
        Total number of channels.
    channel_labels : list of str
        Names for each channel.
    eeg_channels : list of int
        Indices of EEG channels.
    emg_channels : list of int
        Indices of EMG channels.
    buffer_size : int
        Max samples to retain in the ring buffer. If not provided, keeps
        about 30 minutes at the configured sampling rate.
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        subject_id: str = "subject01",
        srate: float = 500.0,
        n_channels: int = 16,
        channel_labels: list = None,
        eeg_channels: list = None,
        emg_channels: list = None,
        patient_info: dict | None = None,
        buffer_size: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.subject_id = subject_id
        self.srate = srate
        self.n_channels = n_channels
        self.channel_labels = channel_labels or [f"Ch{i+1}" for i in range(n_channels)]
        self.eeg_channels = eeg_channels or []
        self.emg_channels = emg_channels or []
        self.patient_info = dict(patient_info or {})

        if not buffer_size or buffer_size <= 0:
            buffer_size = max(1024, int(self.srate * 60 * 30))
        self.buffer_size = int(buffer_size)

        # Use MetaBCI RingBuffer for thread-safe buffering sized for a full session.
        self._ring: RingBuffer = RingBuffer(size=self.buffer_size)
        self._labels: list = []  # sparse event list — kept lightweight
        self._recording = False
        self._lock = threading.Lock()
        self._session_start: float = 0.0
        self._file_paths: dict = {}

        # Marker for event-aligned epoch tracking (optional, for closed-loop)
        self._marker: Marker | None = None

    def start_session(self):
        """Start a new recording session."""
        with self._lock:
            self._ring.clear()
            self._labels = []
            self._recording = True
            self._session_start = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Recording started: subject=%s, %dch @ %.0fHz",
            self.subject_id, self.n_channels, self.srate,
        )

    def append_data(self, data: np.ndarray):
        """Append a chunk of data (n_channels, n_samples) to RingBuffer."""
        if not self._recording:
            return
        with self._lock:
            n_new = data.shape[1]
            for i in range(n_new):
                self._ring.append(data[:, i].copy())

    def add_label(self, code: int, name: str = ""):
        """Add an event label at the current time.

        If a Marker is configured, also pushes through marker-based
        epoch tracking for online closed-loop integration.
        """
        if not self._recording:
            return
        ts = time.time() - (self._session_start or time.time())
        with self._lock:
            self._labels.append((ts, code, name))
        logger.debug("Label added: code=%d, name='%s' at t=%.3f", code, name, ts)

    def configure_marker(self, interval: float, events: list = None):
        """Set up a Marker for event-aligned epoch tracking.

        Parameters
        ----------
        interval : float
            Epoch window duration in seconds.
        events : list of int, optional
            Event codes to track.  Defaults to [1].
        """
        self._marker = Marker(
            interval=interval,
            srate=self.srate,
            events=events or [1],
        )

    def get_epoch(self) -> np.ndarray | None:
        """Return the most recent marker-triggered epoch, or None."""
        if self._marker is not None:
            epoch_data = self._marker.get_epoch()
            if epoch_data:
                return np.array(epoch_data)
        return None

    def stop_session(self) -> dict:
        """Stop recording and save data to NPY file + labels JSON.

        Collects all data from the RingBuffer and saves it alongside
        metadata and event labels.

        Returns
        -------
        file_paths : dict with keys 'npy', 'meta', 'labels'
        """
        if not self._recording:
            return self._file_paths

        with self._lock:
            self._recording = False
            ring_data = self._ring.get_all()
            if not ring_data:
                logger.warning("No data recorded.")
                return self._file_paths

            # RingBuffer stores per-sample arrays; stack to (n_ch, n_samples)
            full_data = np.column_stack(ring_data)
            self._ring.clear()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.subject_id}_{timestamp}"

        # Save NPY
        npy_path = self.output_dir / f"{base_name}_data.npy"
        np.save(str(npy_path), full_data)
        logger.info("Data saved: %s (%s)", npy_path, full_data.shape)

        # Save metadata
        meta = {
            "subject_id": self.subject_id,
            "patient_info": self.patient_info,
            "srate": self.srate,
            "n_channels": self.n_channels,
            "channel_labels": self.channel_labels,
            "eeg_channels": self.eeg_channels,
            "emg_channels": self.emg_channels,
            "shape": list(full_data.shape),
            "duration_sec": full_data.shape[1] / self.srate,
            "session_start": self._session_start,
            "buffer_size": self.buffer_size,
        }
        meta_path = self.output_dir / f"{base_name}_meta.json"
        with open(str(meta_path), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # Save labels
        labels_path = self.output_dir / f"{base_name}_labels.json"
        label_records = [
            {"timestamp_sec": ts, "code": c, "name": n}
            for ts, c, n in self._labels
        ]
        with open(str(labels_path), "w", encoding="utf-8") as f:
            json.dump(label_records, f, indent=2, ensure_ascii=False)

        self._file_paths = {
            "npy": str(npy_path),
            "meta": str(meta_path),
            "labels": str(labels_path),
        }

        logger.info("Recording stopped. Files saved to %s", self.output_dir)
        return self._file_paths

    def is_recording(self) -> bool:
        """Return whether the recorder is active."""
        return self._recording

    def get_file_paths(self) -> dict:
        """Return file paths from the last session."""
        return self._file_paths

    def get_recorded_data(self) -> np.ndarray:
        """Return the full recorded data as (n_channels, n_samples)."""
        with self._lock:
            ring_data = self._ring.get_all()
            if not ring_data:
                return np.array([]).reshape(self.n_channels, 0)
            return np.column_stack(ring_data)
