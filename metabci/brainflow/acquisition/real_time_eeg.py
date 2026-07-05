#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Real-Time EEG/EMG/ECG Visualization — Unified GUI for BrainFlow, TCP amplifiers,
LSL devices, and online processing (ProcessWorker).

Supported amplifiers (13 total):
  BrainFlow — Synthetic, Cyton, Cyton+Daisy, Ganglion (and WiFi variants)
  TCP        — NeuroScan, Curry8, Neuracle, HTOnline Digital EEG
  WiFiShield — OpenBCI WiFi Shield (raw socket, HTTP+TCP)
  LSL        — Generic LSL-compatible devices

Online processing:
  Add custom ProcessWorker subclasses via the GUI.  Each worker receives
  marker-triggered epochs and runs pre → consume → post in a subprocess.

Usage:
  python real_time_eeg.py --amp synthetic
  python real_time_eeg.py --amp neuroscan --host 192.168.1.100 --port 4000
  python real_time_eeg.py --amp lsl
"""

from __future__ import annotations

import argparse
import importlib
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

# ---------------------------------------------------------------------------
# GUI imports
# ---------------------------------------------------------------------------
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGroupBox, QLabel, QComboBox, QPushButton, QSpinBox, QStatusBar,
        QSplitter, QFrame, QStackedWidget, QLineEdit, QToolButton,
        QScrollArea, QSizePolicy, QDialog, QDialogButtonBox, QFormLayout,
        QCheckBox, QDoubleSpinBox,
    )
    from PySide6.QtCore import QTimer, Qt, Signal
    from PySide6.QtGui import QFont, QColor, QPalette
except ImportError as e:
    sys.exit(f"PySide6 is required. Install with: pip install PySide6\nOriginal error: {e}")

try:
    import pyqtgraph as pg
except ImportError:
    sys.exit("pyqtgraph is required. Install with: pip install pyqtgraph")

# ---------------------------------------------------------------------------
# BrainFlow import (optional — only needed for BrainFlow boards)
# ---------------------------------------------------------------------------
try:
    from brainflow.board_shim import (
        BoardShim, BoardIds, BrainFlowInputParams, LogLevels,
    )
    _HAS_BRAINFLOW = True
except ImportError:
    BoardShim = None  # type: ignore
    BoardIds = None   # type: ignore
    BrainFlowInputParams = None
    LogLevels = None
    _HAS_BRAINFLOW = False

# ---------------------------------------------------------------------------
# MetaBCI brainflow imports
# ---------------------------------------------------------------------------
import os as _os
_this_dir = _os.path.dirname(_os.path.abspath(__file__))

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from metabci.brainflow.amplifiers import Marker
from metabci.brainflow.logger import get_logger

logger = get_logger("real_time_eeg")

# WiFiShieldAmplifier — provided by metabci.brainflow.acquisition.sources
try:
    from metabci.brainflow.acquisition.sources import WiFiShieldAmplifier
    _HAS_WIFI_SHIELD = True
except ImportError:
    WiFiShieldAmplifier = None  # type: ignore[assignment]
    _HAS_WIFI_SHIELD = False
    logger.debug("WiFiShieldAmplifier not available (sources module not yet integrated)")


# ============================================================================
# Section 1 — Constants & Configuration
# ============================================================================

# ============================================================
# 通道配置 — 按硬件索引明确指定每个组的通道
# 参照 metabci_EEG/apps/ui/eeg_display.py 的分配方式
# 用户可根据实际电极接线修改以下列表
# ============================================================
ALL_CHANNEL_NAMES = [
    "CH1", "CH2", "CH3", "CH4", "CH5", "CH6",
    "CH7", "CH8", "CH9", "CH10", "CH11", "CH12",
    "CH13", "CH14", "CH15", "CH16",
]

# 硬件通道索引 (0-based)
EEG_CHANNELS = [0, 1, 2, 3, 4, 5, 10, 11, 13]   # 9 EEG 通道
EMG_CHANNELS = [6, 7, 8, 9, 14, 15]              # 6 EMG 通道
ECG_CHANNELS = [12]                                # 1 ECG 通道

# 各组显示参数
REGION_CFG = {
    "EEG": {"trace_spacing_uv": 280.0, "display_target_uv": 50.0,
            "gain_multiplier": 1.0, "channels": EEG_CHANNELS},
    "EMG": {"trace_spacing_uv": 220.0, "display_target_uv": 42.0,
            "gain_multiplier": 0.25, "channels": EMG_CHANNELS},
    "ECG": {"trace_spacing_uv": 260.0, "display_target_uv": 42.0,
            "gain_multiplier": 0.25, "channels": ECG_CHANNELS},
}

TRACE_COLORS: List[str] = [
    "#00FFEA", "#FF4444", "#FFDD00", "#D47FFF",
    "#33B5FF", "#FF9500", "#00FF55", "#FF3366",
    "#00CCFF", "#FF66AA", "#55FF99", "#FFCC00",
    "#CC66FF", "#0099FF", "#00FFAA", "#FF6633",
]

GROUP_PRIORITY = ["EEG", "EMG", "ECG"]

DEFAULT_SAMPLE_RATE: float = 250.0
DEFAULT_BUFFER_SECS: float = 15.0
DEFAULT_VISIBLE_SECS: float = 8.0
REFRESH_INTERVAL_MS: int = 33
DOWNSAMPLE_THRESHOLD: int = 1200
ARTIFACT_SIGMA: float = 6.0
MAX_HARDWARE_CHANNELS: int = 16

# ---------------------------------------------------------------------------
# TCP amplifier lifecycle descriptors
# ---------------------------------------------------------------------------
LIFECYCLE: Dict[str, Dict[str, List[str]]] = {
    "NeuroScan": {
        "connect":    ["connect_tcp"],
        "start":      ["start_acq", "start_trans"],
        "stop":       ["stop_trans", "stop_acq"],
        "disconnect": ["close_connection"],
    },
    "Curry8": {
        "connect":    ["connect_tcp"],
        "start":      ["start_acq", "start_trans"],
        "stop":       ["stop_trans", "stop_acq"],
        "disconnect": ["close_connection"],
    },
    "Neuracle": {
        "connect":    ["connect_tcp"],
        "start":      ["start_trans"],
        "stop":       ["stop_trans"],
        "disconnect": ["close_connection"],
    },
    "HTOnline": {
        "connect":    ["connect_tcp"],
        "start":      ["start_acq"],
        "stop":       ["stop_acq"],
        "disconnect": ["close_connection"],
    },
}

# ---------------------------------------------------------------------------
# Unified amplifier registry — drives dropdown, config panel, factory
# ---------------------------------------------------------------------------
def _bf_board_ids():
    """Return a safe mapping of BrainFlow board name → BoardId enum."""
    if not _HAS_BRAINFLOW or BoardIds is None:
        return {}
    return {
        "synthetic":            BoardIds.SYNTHETIC_BOARD,
        "cyton":                BoardIds.CYTON_BOARD,
        "cyton_daisy":          BoardIds.CYTON_DAISY_BOARD,
        "ganglion":             BoardIds.GANGLION_BOARD,
        "cyton_wifi":           BoardIds.CYTON_WIFI_BOARD,
        "cyton_daisy_wifi":     BoardIds.CYTON_DAISY_WIFI_BOARD,
        "ganglion_wifi":        BoardIds.GANGLION_WIFI_BOARD,
    }

def _build_registry() -> List[Dict[str, Any]]:
    """Build the amplifier registry list — WiFi Shield only."""
    reg: List[Dict[str, Any]] = []

    # -- WiFi Shield (OpenBCI raw socket) ---------------------------------
    reg.append({
        "key": "wifi_shield", "cat": "wifi_shield",
        "label": "OpenBCI WiFi Shield (Raw Socket — 16ch)",
        "needs_host": True, "needs_port": True,
        "default_host": "192.168.4.1", "default_port": 9000,
        "default_srate": 500, "default_nch": 16,
        "available": _HAS_WIFI_SHIELD,
    })

    return reg

AMPLIFIER_REGISTRY: List[Dict[str, Any]] = _build_registry()


# ============================================================================
# Section 2 — SignalProcessor (enhanced)
# ============================================================================

class SignalProcessor:
    """单处理器 — 全部通道统一滤波（参照 eeg_display.py）。

    复刻 stroke 项目的 RealtimeEEGProcessor：
    - 每通道独立 EMA 去基线（1 秒时间常数）
    - 共用 1-30 Hz 带通 + 50 Hz 陷波（合并 SOS）
    - 每通道独立 IIR 状态向量，跨批次连续
    """

    def __init__(self, channels: int, sample_rate: float,
                 low_hz: float = 1.0, high_hz: float = 30.0,
                 notch_hz: float = 50.0,
                 baseline_seconds: float = 1.0) -> None:
        self.channels = int(channels)
        self.sample_rate = float(sample_rate)
        self._notch_hz = float(notch_hz)
        self._low_hz = float(low_hz)
        self._high_hz = float(high_hz)

        # -- 构建合并的 SOS 滤波器 (带通 + 陷波) -------------------------------
        self._all_sos: List[np.ndarray] = []
        nyquist = self.sample_rate / 2.0
        high = min(high_hz, nyquist * 0.90)
        if low_hz > 0 and high > low_hz:
            sos_bp = butter(4, [low_hz, high], btype="bandpass",
                            fs=self.sample_rate, output="sos")
            self._all_sos.append(sos_bp)
        if notch_hz > 0:
            w0 = notch_hz / nyquist
            Q = 30.0
            sos_notch = butter(2, [w0 - w0 / Q, w0 + w0 / Q],
                               btype="bandstop", output="sos")
            self._all_sos.append(sos_notch)

        if self._all_sos:
            combined = self._all_sos[0]
            for s in self._all_sos[1:]:
                combined = np.vstack([combined, s])
            self.sos = combined
        else:
            self.sos = None

        self.zi = np.zeros((self.sos.shape[0], 2, self.channels),
                           dtype=np.float64) if self.sos is not None else None

        # -- 每通道 EMA 去基线状态 ------------------------------------------------
        self.baseline = np.zeros((self.channels,), dtype=np.float64)
        self.baseline_ready = np.zeros((self.channels,), dtype=bool)
        self.baseline_alpha = 1.0 / max(1.0, sample_rate * baseline_seconds)

    # ------------------------------------------------------------------
    # 对外配置接口（兼容原有调用）
    # ------------------------------------------------------------------

    def set_line_freq(self, freq: float) -> None:
        """运行时切换工频频率并重建陷波器。"""
        self._notch_hz = float(freq)
        self._rebuild_filters()

    def set_filter_order(self, order: int) -> None:
        """运行时切换带通滤波阶数。"""
        self._rebuild_filters(order=max(2, min(10, int(order))))

    def set_artifact_sigma(self, sigma: float) -> None:
        """保留接口（当前处理器不做伪迹剔除）。"""
        pass

    def set_baseline_tau(self, seconds: float) -> None:
        """更改 EMA 基线时间常数。"""
        self.baseline_alpha = 1.0 / max(1.0, self.sample_rate * max(0.1, seconds))

    def _rebuild_filters(self, order: int = 4) -> None:
        """重建带通 + 陷波 SOS 及 zi 状态。"""
        self._all_sos.clear()
        nyquist = self.sample_rate / 2.0
        if self._low_hz > 0 and self._high_hz > self._low_hz:
            sos_bp = butter(order, [self._low_hz, self._high_hz],
                            btype="bandpass", fs=self.sample_rate, output="sos")
            self._all_sos.append(sos_bp)
        if self._notch_hz > 0:
            w0 = self._notch_hz / nyquist
            Q = 30.0
            sos_notch = butter(2, [w0 - w0 / Q, w0 + w0 / Q],
                               btype="bandstop", output="sos")
            self._all_sos.append(sos_notch)
        if self._all_sos:
            combined = self._all_sos[0]
            for s in self._all_sos[1:]:
                combined = np.vstack([combined, s])
            self.sos = combined
        else:
            self.sos = None
        self.zi = np.zeros((self.sos.shape[0], 2, self.channels),
                           dtype=np.float64) if self.sos is not None else None

    # ------------------------------------------------------------------
    # 主处理 — 复刻 eeg_display.py 的 process_batch
    # ------------------------------------------------------------------

    def process(self, data: np.ndarray,
                eeg_all: Optional[np.ndarray] = None) -> np.ndarray:
        """处理数据块：(n_channels, n_samples) → (n_channels, n_samples)。"""
        if data.size == 0:
            return data
        # data 形状 (n_channels, n_samples)，转置为 (n_samples, n_channels)
        src = data.T.astype(np.float64, copy=False)
        n_samples = src.shape[0]
        n_ch = min(src.shape[1], self.channels)

        values = np.full((n_samples, self.channels), np.nan, dtype=np.float64)
        if n_ch:
            values[:, :n_ch] = src[:, :n_ch]

        detrended = np.zeros((n_samples, self.channels), dtype=np.float64)
        valid_mask = np.isfinite(values)

        # -- 1. 逐样本 EMA 去基线 (每通道独立) --------------------------------
        for i in range(n_samples):
            row_valid = valid_mask[i]
            if not np.any(row_valid):
                continue
            first_valid = row_valid & ~self.baseline_ready
            if np.any(first_valid):
                self.baseline[first_valid] = values[i, first_valid]
                self.baseline_ready[first_valid] = True
            ready_valid = row_valid & self.baseline_ready
            if np.any(ready_valid):
                cur = values[i, ready_valid]
                self.baseline[ready_valid] += self.baseline_alpha * (
                    cur - self.baseline[ready_valid])
                detrended[i, ready_valid] = cur - self.baseline[ready_valid]

        # -- 2. SOS 滤波 (带通 + 陷波) -----------------------------------------
        if self.sos is not None and self.zi is not None:
            filtered, self.zi = sosfilt(
                self.sos, detrended, axis=0, zi=self.zi)
        else:
            filtered = detrended

        filtered[~valid_mask] = np.nan

        # -- 3. CAR (可选，利用 eeg_all 通道做共同平均参考) -------------------
        if eeg_all is not None and eeg_all.shape[0] > 1:
            car_mean = np.nanmean(eeg_all, axis=0)
            filtered = filtered - car_mean[np.newaxis, :]

        # 转回 (n_channels, n_samples)
        return filtered.T.astype(np.float64, copy=False)

    def reset(self) -> None:
        """重置滤波器状态和基线。"""
        self.baseline = np.zeros((self.channels,), dtype=np.float64)
        self.baseline_ready = np.zeros((self.channels,), dtype=bool)
        if self.sos is not None:
            self.zi = np.zeros((self.sos.shape[0], 2, self.channels),
                               dtype=np.float64)


# ============================================================================
# Section 3 — _ChannelGroup (unchanged)
# ============================================================================

class _ChannelGroup:
    """一组通道在一个 PlotWidget 内用垂直偏移堆叠显示。

    参照 eeg_display.py 的设计：
    - channel_idxs 指定硬件通道索引
    - 用 ordered_data[ch_idx] 直接取数据
    - Y 轴刻度 = 通道名，位置与偏移量一一对应
    """

    def __init__(self, plot_item: pg.PlotItem,
                 channel_idxs: List[int],
                 channel_names: List[str],
                 colors: List[str],
                 trace_spacing_uv: float = 160.0,
                 display_target_uv: float = 42.0,
                 gain_multiplier: float = 1.0,
                 max_draw_points: int = 1200) -> None:
        self.channel_idxs = list(channel_idxs)
        self.n_ch = len(channel_idxs)
        self._names = list(channel_names)
        self._spacing = trace_spacing_uv
        self._target = display_target_uv
        self._gain_mul = gain_multiplier
        self._max_pts = max_draw_points
        self._soft_limit = 72.0
        self._min_gain = 0.02 * gain_multiplier
        self._max_gain = 5.0 * gain_multiplier

        # 垂直偏移 — 最上面通道偏移最大
        self._offsets = np.asarray([
            (self.n_ch - 1 - i) * trace_spacing_uv
            for i in range(self.n_ch)
        ], dtype=np.float32)

        # 清除旧内容，设置轴
        plot_item.clear()
        plot_item.showGrid(x=True, y=True, alpha=0.15)
        plot_item.setMouseEnabled(x=True, y=False)
        plot_item.hideButtons()
        plot_item.setMenuEnabled(False)
        plot_item.getViewBox().setBackgroundColor((18, 18, 22))

        # X-axis
        plot_item.getAxis("bottom").setPen(pg.mkPen(color="#777"))
        plot_item.getAxis("bottom").setTextPen(pg.mkPen(color="#aaa"))
        plot_item.getAxis("bottom").setTickFont(QFont("Segoe UI", 9))

        # Y-axis — 刻度 = 通道名，位置与偏移一一对应
        y_ticks = [
            (float(self._offsets[i]), f"  {channel_names[i]}")
            for i in range(self.n_ch)
        ]
        y_axis = plot_item.getAxis("left")
        y_axis.setTicks([y_ticks])
        y_axis.setPen(pg.mkPen(color="#777"))
        y_axis.setTextPen(pg.mkPen(color="#ffffff", width=1.0))
        y_axis.setTickFont(QFont("Segoe UI", 12, QFont.Bold))
        y_axis.setStyle(showValues=True)

        # Y 范围
        plot_item.setYRange(
            -trace_spacing_uv,
            float(self._offsets[0] + trace_spacing_uv),
            padding=0.02,
        )

        self._plot_item = plot_item
        self._curves: List[pg.PlotDataItem] = []
        for i in range(self.n_ch):
            pen = pg.mkPen(colors[i % len(colors)], width=1.1)
            curve = plot_item.plot(pen=pen, connect="finite", antialias=True)
            curve.setClipToView(True)
            curve.setDownsampling(auto=False)
            self._curves.append(curve)

    # ------------------------------------------------------------------
    # 显示更新
    # ------------------------------------------------------------------

    def update_plot(self, ordered_data: np.ndarray, fs: float,
                    window_s: float) -> None:
        """更新曲线。

        Parameters
        ----------
        ordered_data : ndarray (n_all_channels, n_samples)
            按硬件通道索引排列的数据
        fs : 采样率
        window_s : 显示窗口秒数
        """
        n = ordered_data.shape[1]
        if n == 0:
            return

        # 时间轴 — 参照 eeg_display.py
        if n <= 1:
            x = np.asarray([window_s], dtype=np.float32)
        else:
            duration = (n - 1) / fs
            shift = max(0.0, window_s - duration)
            x = shift + (np.arange(n, dtype=np.float32) / fs)

        # 降采样
        if n > self._max_pts:
            step = int(np.ceil(n / self._max_pts))
            indices = np.arange(0, n, step, dtype=np.int32)
            if indices[-1] != n - 1:
                indices = np.append(indices, n - 1)
            x = x[indices]
            ordered_data = ordered_data[:, indices]

        # 更新曲线 — 用硬件通道索引直接取数据
        for i, (ch_idx, curve) in enumerate(
            zip(self.channel_idxs, self._curves)
        ):
            if ch_idx >= ordered_data.shape[0]:
                continue
            y = self._scale_trace(ordered_data[ch_idx])
            curve.setData(x, y + self._offsets[i], connect="finite")

        # 更新 X 轴范围 (参照 eeg_display.py set_visible_duration)
        self._plot_item.setXRange(0.0, window_s, padding=0.0)

    def _scale_trace(self, values_uv: np.ndarray) -> np.ndarray:
        """自适应增益 + 线性裁剪（去掉 tanh 避免平坦/梯形失真）。"""
        y = np.asarray(values_uv, dtype=np.float32).copy()
        finite = np.isfinite(y)
        if not np.any(finite):
            return y
        yf = y[finite]
        center = float(np.median(yf))
        yf = yf - center
        dev = np.abs(yf)
        amp95 = float(np.percentile(dev, 95))
        if np.isfinite(amp95) and amp95 > 1e-6:
            gain = self._target / amp95
            gain = float(np.clip(gain, self._min_gain, self._max_gain))
        else:
            gain = 1.0
        yf = yf * gain
        # 线性裁剪到 ±soft_limit，不产生曲线变形
        yf = np.clip(yf, -self._soft_limit, self._soft_limit)
        y[finite] = yf
        return y

    # ------------------------------------------------------------------
    # 缩放 & 清理
    # ------------------------------------------------------------------

    def set_gain(self, multiplier: float) -> None:
        """设置增益倍率用于垂直缩放 (0.05 ~ 10.0)。"""
        self._gain_mul = float(np.clip(multiplier, 0.05, 10.0))
        self._min_gain = 0.02 * self._gain_mul
        self._max_gain = 5.0 * self._gain_mul

    def clear(self) -> None:
        for c in self._curves:
            c.clear()


# ============================================================================
# Section 4 — AbstractAmplifier & Adapters
# ============================================================================

class AbstractAmplifier(ABC):
    """Unified interface for all data sources."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def start_acquisition(self) -> None: ...

    @abstractmethod
    def get_data(self) -> np.ndarray:
        """Return (total_rows, n_new) numpy array. Returns empty (0,0) if no data."""

    @abstractmethod
    def stop_acquisition(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def sample_rate(self) -> float: ...

    @property
    @abstractmethod
    def channel_count(self) -> int:
        """Number of data channels (excluding trigger)."""

    @property
    @abstractmethod
    def channel_names(self) -> List[str]: ...

    @property
    def has_trigger(self) -> bool:
        return False

    @property
    def trigger_row(self) -> int:
        """Row index of the trigger column in get_data() output, or -1."""
        return -1


# ---------------------------------------------------------------------------
# BrainFlowAdapter
# ---------------------------------------------------------------------------

class BrainFlowAdapter(AbstractAmplifier):
    """Wraps a BrainFlow BoardShim session."""

    def __init__(self, board_id: int, serial_port: str = "",
                 ip_address: str = "", ip_port: int = 0) -> None:
        if not _HAS_BRAINFLOW:
            raise RuntimeError("brainflow package is not installed.")
        self._board_id = board_id
        self._serial_port = serial_port
        self._ip_address = ip_address
        self._ip_port = ip_port
        self._board: Any = None
        self._fs: float = DEFAULT_SAMPLE_RATE
        # Channel layout (populated in connect)
        self._ch_groups: Dict[str, Dict] = {}
        self._row_to_idx: Dict[int, int] = {}
        self._n_data_ch: int = 0

    # -- AbstractAmplifier interface ----------------------------------------

    def connect(self) -> None:
        params = BrainFlowInputParams()
        params.serial_port = self._serial_port
        params.ip_address = self._ip_address
        params.ip_port = self._ip_port
        self._board = BoardShim(self._board_id, params)
        BoardShim.enable_board_logger()
        BoardShim.set_log_level(LogLevels.LEVEL_INFO.value)
        self._board.prepare_session()
        self._fs = float(self._board.get_sampling_rate(self._board_id))
        # Discover channel layout
        self._ch_groups, self._n_data_ch = _discover_channel_layout(self._board_id)
        self._row_to_idx.clear()
        idx = 0
        for gname in GROUP_PRIORITY:
            info = self._ch_groups.get(gname)
            if info is None:
                continue
            for row in info["rows"]:
                self._row_to_idx[row] = idx
                idx += 1

    def start_acquisition(self) -> None:
        if self._board is not None:
            self._board.start_stream()

    def get_data(self) -> np.ndarray:
        if self._board is None:
            return np.empty((0, 0))
        raw = self._board.get_board_data()
        if raw.shape[1] == 0:
            return np.empty((0, 0))
        # Map BrainFlow rows → flat data array
        n_new = raw.shape[1]
        out = np.zeros((self._n_data_ch, n_new), dtype=np.float64)
        for bf_row, buf_idx in self._row_to_idx.items():
            if bf_row < raw.shape[0]:
                out[buf_idx, :] = raw[bf_row, :].astype(np.float64)
        return out

    def stop_acquisition(self) -> None:
        if self._board is not None:
            try:
                self._board.stop_stream()
            except Exception:
                pass

    def disconnect(self) -> None:
        if self._board is not None:
            try:
                self._board.release_session()
            except Exception:
                pass
            self._board = None

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def channel_count(self) -> int:
        return self._n_data_ch

    @property
    def channel_names(self) -> List[str]:
        names: List[str] = []
        for gname in GROUP_PRIORITY:
            info = self._ch_groups.get(gname)
            if info:
                names.extend(info.get("names", []))
        return names

    @property
    def has_trigger(self) -> bool:
        return False

    @property
    def channel_groups(self) -> Dict[str, Dict]:
        return self._ch_groups


# ---------------------------------------------------------------------------
# TCPAmplifierAdapter
# ---------------------------------------------------------------------------

class TCPAmplifierAdapter(AbstractAmplifier):
    """Generic adapter for NeuroScan / Curry8 / Neuracle / HTOnline.

    Runs a background collection thread that calls *amp.recv()* and pushes
    samples into a thread-safe deque.
    """

    def __init__(self, amp: Any, lifecycle_key: str,
                 num_chans: int, srate: float) -> None:
        self._amp = amp
        self._lifecycle = LIFECYCLE.get(lifecycle_key, {})
        self._n_ch = int(num_chans)
        self._fs = float(srate)
        self._deque: deque = deque()
        self._lock = threading.Lock()
        self._exit = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- AbstractAmplifier interface ----------------------------------------

    def connect(self) -> None:
        for method_name in self._lifecycle.get("connect", []):
            getattr(self._amp, method_name)()

    def start_acquisition(self) -> None:
        # Call amplifier-specific start sequence
        for method_name in self._lifecycle.get("start", []):
            getattr(self._amp, method_name)()
        # Start collection thread
        self._exit.clear()
        self._thread = threading.Thread(target=self._collect_loop,
                                        name="tcp-collect", daemon=True)
        self._thread.start()

    def get_data(self) -> np.ndarray:
        with self._lock:
            if not self._deque:
                return np.empty((0, 0))
            n = len(self._deque)
            samples = [self._deque.popleft() for _ in range(n)]
        if not samples:
            return np.empty((0, 0))
        # samples is a list of lists: [[ch1,...,chN,trigger], ...]
        arr = np.array(samples, dtype=np.float64).T  # (n_ch+1, n_samples)
        return arr

    def stop_acquisition(self) -> None:
        self._exit.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for method_name in self._lifecycle.get("stop", []):
            try:
                getattr(self._amp, method_name)()
            except Exception:
                pass

    def disconnect(self) -> None:
        if self._exit.is_set() or self._thread is None:
            pass  # already stopped
        else:
            self.stop_acquisition()
        for method_name in self._lifecycle.get("disconnect", []):
            try:
                getattr(self._amp, method_name)()
            except Exception:
                pass

    def _collect_loop(self) -> None:
        while not self._exit.is_set():
            try:
                samples = self._amp.recv()
            except Exception:
                time.sleep(0.001)
                continue
            if samples:  # handles None and []
                with self._lock:
                    self._deque.extend(samples)
            else:
                time.sleep(0.001)

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def channel_count(self) -> int:
        return self._n_ch

    @property
    def channel_names(self) -> List[str]:
        # Try to discover channel names if the amplifier supports it
        try:
            if hasattr(self._amp, "update_channel_info"):
                self._amp.update_channel_info()
                if hasattr(self._amp, "chanelNameList"):
                    return self._amp.chanelNameList[:self._n_ch]
            if hasattr(self._amp, "get_name_chans"):
                return self._amp.get_name_chans()[:self._n_ch]
        except Exception:
            pass
        return [f"Ch{i + 1}" for i in range(self._n_ch)]

    @property
    def has_trigger(self) -> bool:
        return True

    @property
    def trigger_row(self) -> int:
        return self._n_ch  # trigger is the last column


# ---------------------------------------------------------------------------
# LSLAdapter
# ---------------------------------------------------------------------------

class LSLAdapter(AbstractAmplifier):
    """Wraps LSLapps for generic LSL-compatible devices."""

    STREAM_TIMEOUT = 15.0  # seconds to wait for LSL stream discovery

    def __init__(self) -> None:
        self._lsl: Optional[Any] = None
        self._fs: float = DEFAULT_SAMPLE_RATE
        self._n_ch: int = 0
        self._names: List[str] = []
        self._deque: deque = deque()
        self._lock = threading.Lock()
        self._exit = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def connect(self) -> None:
        self._lsl = LSLapps()
        # Wait for stream discovery
        deadline = time.time() + self.STREAM_TIMEOUT
        while time.time() < deadline:
            if self._lsl.data_inlet is not None:
                break
            time.sleep(0.3)
        else:
            raise RuntimeError(
                f"No LSL data stream found within {self.STREAM_TIMEOUT:.0f} seconds. "
                "Make sure an LSL outlet is active."
            )
        # Gather stream info
        info = self._lsl.data_inlet.inlet.info()
        self._fs = float(info.nominal_srate())
        self._n_ch = int(info.channel_count())
        # Try to get channel names from stream info
        ch_xml = info.desc().child("channels")
        if not ch_xml.empty():
            self._names = [ch.child_value("label") or f"Ch{i + 1}"
                           for i, ch in enumerate(ch_xml.child("channel"))]
        if len(self._names) < self._n_ch:
            self._names = [f"Ch{i + 1}" for i in range(self._n_ch)]

    def start_acquisition(self) -> None:
        self._exit.clear()
        self._thread = threading.Thread(target=self._collect_loop,
                                        name="lsl-collect", daemon=True)
        self._thread.start()

    def get_data(self) -> np.ndarray:
        with self._lock:
            if not self._deque:
                return np.empty((0, 0))
            n = len(self._deque)
            samples = [self._deque.popleft() for _ in range(n)]
        if not samples:
            return np.empty((0, 0))
        arr = np.array(samples, dtype=np.float64).T
        return arr

    def stop_acquisition(self) -> None:
        self._exit.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def disconnect(self) -> None:
        self.stop_acquisition()
        self._lsl = None

    def _collect_loop(self) -> None:
        while not self._exit.is_set():
            if self._lsl is None:
                break
            try:
                samples = self._lsl.recv()
            except Exception:
                time.sleep(0.005)
                continue
            if samples:
                with self._lock:
                    self._deque.extend(samples)
            else:
                time.sleep(0.005)

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def channel_count(self) -> int:
        return self._n_ch

    @property
    def channel_names(self) -> List[str]:
        return self._names

    @property
    def has_trigger(self) -> bool:
        return True

    @property
    def trigger_row(self) -> int:
        return self._n_ch


# ---------------------------------------------------------------------------
# WiFiShieldAdapter — raw socket protocol for OpenBCI WiFi Shield
# ---------------------------------------------------------------------------

class WiFiShieldAdapter(AbstractAmplifier):
    """Wraps the metabci_EEG WiFiShieldAmplifier (HTTP + raw TCP protocol).

    This is the recommended adapter for OpenBCI Cyton / Cyton+Daisy with
    WiFi Shield.  It uses HTTP (port 80) for control and a local TCP server
    (default port 9000) for streaming — no BrainFlow SDK needed.
    """

    def __init__(self, host: str = "192.168.4.1", port: int = 9000,
                 n_channels: int = 16, srate: float = 500.0) -> None:
        if not _HAS_WIFI_SHIELD:
            raise RuntimeError(
                "WiFiShieldAmplifier is not available. "
                "Make sure the metabci_EEG project is accessible."
            )
        self._host = host
        self._port = int(port)
        self._n_ch = int(n_channels)
        self._fs = float(srate)
        self._samples_read: int = 0
        self._amp: Any = None
        # Per-channel EMA baseline tracker (launcher-style: α=1/(fs*1s))
        self._baseline: Optional[np.ndarray] = None
        _bl_alpha: float = 0.0  # set in start_acquisition

    # -- AbstractAmplifier interface ----------------------------------------

    def connect(self) -> None:
        pass

    def start_acquisition(self) -> None:
        self._amp = WiFiShieldAmplifier(
            host=self._host, port=self._port,
            n_channels=self._n_ch, srate=self._fs,
        )
        self._amp.start()
        self._samples_read = 0
        # ~1 second time constant (matches launcher: baseline_seconds=1.0)
        self._bl_alpha = 1.0 / max(1.0, self._fs * 1.0)
        self._baseline = None

    def get_data(self) -> np.ndarray:
        if self._amp is None or not self._amp.is_streaming():
            return np.empty((0, 0))
        total = self._amp.get_sample_count()
        n_new = total - self._samples_read
        if n_new <= 0:
            return np.empty((0, 0))
        data = self._amp.get_recent(max(n_new, 1))
        if data is None:
            return np.empty((0, 0))
        self._samples_read = total
        result = data.astype(np.float64)  # (n_ch, n_new)

        # Launcher-style baseline removal: EMA per channel, sample by sample.
        # This removes the ~±700k ADC DC offset BEFORE the GUI's bandpass
        # filter sees the data — exactly what the working launcher does.
        if self._baseline is None:
            # Use per-channel median for instant DC convergence
            # (was: first-sample copy — took ~5 s to settle)
            self._baseline = np.median(result, axis=1)
        a = self._bl_alpha
        for ch in range(result.shape[0]):
            bl = self._baseline[ch]
            row = result[ch, :]
            for i in range(len(row)):
                val = row[i]
                bl += a * (val - bl)
                row[i] = val - bl
            self._baseline[ch] = bl
        return result

    def stop_acquisition(self) -> None:
        if self._amp is not None:
            try:
                self._amp.stop()
            except Exception:
                pass

    def disconnect(self) -> None:
        self.stop_acquisition()
        self._amp = None

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def channel_count(self) -> int:
        return self._n_ch

    @property
    def channel_names(self) -> List[str]:
        return [f"Ch{i + 1}" for i in range(self._n_ch)]

    @property
    def has_trigger(self) -> bool:
        return False

    @property
    def trigger_row(self) -> int:
        return -1


# ============================================================================
# Section 5 — Channel layout discovery (BrainFlow)
# ============================================================================

def _discover_channel_layout(board_id: int):
    """Query BrainFlow for the actual channel layout of *board_id*."""
    def _get_channels(kind: str) -> List[int]:
        method = getattr(BoardShim, f"get_{kind}_channels", None)
        if method is None:
            return []
        try:
            raw = method(board_id)
        except Exception:
            return []
        return sorted(set(int(r) for r in raw))

    available: Dict[str, List[int]] = {}
    for group_name in GROUP_PRIORITY:
        available[group_name] = _get_channels(group_name.lower())

    used_global: set = set()
    groups_out: Dict[str, Dict] = {}

    # Try to get real electrode names from BrainFlow
    eeg_names: List[str] = []
    try:
        raw_names = BoardShim.get_eeg_names(board_id)
        if raw_names:
            eeg_names = [n.strip() for n in raw_names.split(",") if n.strip()]
    except Exception:
        pass

    for group_name in GROUP_PRIORITY:
        cfg = FILTER_CFG[group_name]
        max_ch = cfg.get("max_channels", 8)
        candidates = [r for r in available.get(group_name, [])
                      if r not in used_global]
        if not candidates:
            eeg_candidates = [r for r in available.get("EEG", [])
                              if r not in used_global]
            candidates = eeg_candidates
        selected = candidates[:max_ch]
        used_global.update(selected)
        if not selected:
            continue
        # Use real 10-20 names when available, fall back to generic
        names = []
        for r in selected:
            idx = r - 1  # 1-based row → 0-based index into eeg_names
            if idx < len(eeg_names) and eeg_names[idx]:
                names.append(eeg_names[idx])
            else:
                names.append(f"{group_name}{r}")
        groups_out[group_name] = {
            "rows": selected, "names": names,
            "n_channels": len(selected),
        }
    n_total = sum(g["n_channels"] for g in groups_out.values())
    return groups_out, n_total


def _build_generic_channel_groups(n_ch: int, ch_names: List[str],
                                   max_ch: int = MAX_HARDWARE_CHANNELS):
    """Build non-overlapping channel groups for non-BrainFlow amplifiers.

    Partitioning (up to 16 display channels):
      - EEG: first 8 channels  (rows 1-8)
      - EMG: next 6 channels   (rows 9-14) — only if ≥14 total
      - ECG: next 1 channel    (row 15)    — only if ≥15 total
    """
    n_show = min(n_ch, max_ch)
    # Pad channel names
    full_names = list(ch_names)
    while len(full_names) < n_show:
        full_names.append(f"Ch{len(full_names) + 1}")

    # Determine per-group counts (use all available channels, max 16)
    eeg_n = min(n_show, 8)
    remaining = n_show - eeg_n
    emg_n = min(remaining - 1, 6) if remaining >= 7 else min(remaining, 6)
    ecg_n = min(n_show - eeg_n - emg_n, 1) if n_show - eeg_n - emg_n >= 1 else 0
    # Ensure we don't lose any channel: add leftover to EEG
    total_assigned = eeg_n + emg_n + ecg_n
    if total_assigned < n_show:
        eeg_n += n_show - total_assigned

    groups: Dict[str, Dict] = {}
    cursor = 0  # 0-based index into channel names

    if eeg_n > 0:
        groups["EEG"] = {
            "rows": list(range(1, eeg_n + 1)),
            "names": full_names[cursor:cursor + eeg_n],
            "n_channels": eeg_n,
        }
        cursor += eeg_n

    if emg_n > 0:
        groups["EMG"] = {
            "rows": list(range(eeg_n + 1, eeg_n + emg_n + 1)),
            "names": full_names[cursor:cursor + emg_n],
            "n_channels": emg_n,
        }
        cursor += emg_n

    if ecg_n > 0:
        groups["ECG"] = {
            "rows": [eeg_n + emg_n + 1],
            "names": [full_names[cursor]],
            "n_channels": 1,
        }

    n_total = eeg_n + emg_n + ecg_n
    return groups, n_total


# ============================================================================
# Section 6 — Amplifier factory
# ============================================================================

def create_amplifier(entry: Dict[str, Any], **overrides) -> AbstractAmplifier:
    """Factory: given a registry entry, return the correct adapter."""
    cat = entry["cat"]
    if cat == "wifi_shield":
        host = overrides.get("host", entry.get("default_host", "192.168.4.1"))
        port = int(overrides.get("port", entry.get("default_port", 9000)))
        srate = float(overrides.get("srate", entry.get("default_srate", 500)))
        nch = int(overrides.get("nch", entry.get("default_nch", 16)))
        return WiFiShieldAdapter(host=host, port=port,
                                  n_channels=nch, srate=srate)
    else:
        raise ValueError(f"不支持的放大器类型: {cat}")


# ============================================================================
# Section 8 — GUI panels: amplifier config
# ============================================================================

class AmplifierConfigWidget(QWidget):
    """WiFi Shield 配置面板。"""

    configChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(QLabel("主机:"))
        self._wifi_host = QLineEdit("192.168.4.1")
        self._wifi_host.setPlaceholderText("例如: 192.168.4.1")
        self._wifi_host.setMinimumWidth(140)
        self._wifi_host.setMaximumWidth(160)
        self._wifi_host.setToolTip("WiFi Shield IP 地址 (HTTP 控制端口 80)")
        self._wifi_host.setStyleSheet(
            "QLineEdit { background-color: #3a3a45; color: #fff; "
            "border: 1px solid #888; border-radius: 3px; padding: 3px 6px; font-size: 11px; }"
            "QLineEdit:focus { border: 1px solid #4ac0ff; background-color: #2d2d38; }"
        )
        layout.addWidget(self._wifi_host)
        layout.addWidget(QLabel("TCP端口:"))
        self._wifi_port = QSpinBox()
        self._wifi_port.setRange(1024, 65535)
        self._wifi_port.setValue(9000)
        self._wifi_port.setMinimumWidth(70)
        self._wifi_port.setMaximumWidth(80)
        self._wifi_port.setToolTip("Shield 数据流的本地 TCP 端口 (默认: 9000)")
        layout.addWidget(self._wifi_port)
        layout.addWidget(QLabel("通道数:"))
        self._wifi_nch = QComboBox()
        self._wifi_nch.addItems(["8", "16"])
        self._wifi_nch.setCurrentText("16")
        self._wifi_nch.setMaximumWidth(55)
        self._wifi_nch.setStyleSheet(
            "QComboBox { background-color: #3a3a45; color: #fff; "
            "border: 1px solid #888; border-radius: 3px; padding: 3px 6px; font-size: 11px; }"
            "QComboBox:hover { border: 1px solid #4ac0ff; }"
            "QComboBox:focus { border: 1px solid #4ac0ff; background-color: #2d2d38; }"
            "QComboBox::drop-down { border: none; width: 16px; }"
            "QComboBox::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #aaa; margin-right: 4px; }"
            "QComboBox QAbstractItemView { background-color: #2d2d35; color: #fff; selection-background-color: #3a6fc4; border: 1px solid #666; outline: none; }"
        )
        layout.addWidget(self._wifi_nch)
        layout.addWidget(QLabel("采样率:"))
        self._wifi_srate = QComboBox()
        self._wifi_srate.addItems(["250", "500"])
        self._wifi_srate.setCurrentText("500")
        self._wifi_srate.setMaximumWidth(60)
        self._wifi_srate.setStyleSheet(
            "QComboBox { background-color: #3a3a45; color: #fff; "
            "border: 1px solid #888; border-radius: 3px; padding: 3px 6px; font-size: 11px; }"
            "QComboBox:hover { border: 1px solid #4ac0ff; }"
            "QComboBox:focus { border: 1px solid #4ac0ff; background-color: #2d2d38; }"
            "QComboBox::drop-down { border: none; width: 16px; }"
            "QComboBox::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #aaa; margin-right: 4px; }"
            "QComboBox QAbstractItemView { background-color: #2d2d35; color: #fff; selection-background-color: #3a6fc4; border: 1px solid #666; outline: none; }"
        )
        layout.addWidget(self._wifi_srate)
        layout.addStretch()

        # Signals
        self._wifi_host.textChanged.connect(lambda: self.configChanged.emit())
        self._wifi_port.valueChanged.connect(lambda: self.configChanged.emit())
        self._wifi_nch.currentTextChanged.connect(lambda: self.configChanged.emit())
        self._wifi_srate.currentTextChanged.connect(lambda: self.configChanged.emit())

    def show_for_category(self, cat: str, entry: Dict[str, Any]) -> None:
        """Populate defaults (always WiFi Shield)."""
        self._wifi_host.setText(entry.get("default_host", "192.168.4.1"))
        self._wifi_port.setValue(entry.get("default_port", 9000))
        self._wifi_nch.setCurrentText(str(entry.get("default_nch", 16)))
        self._wifi_srate.setCurrentText(str(entry.get("default_srate", 500)))

    def get_config(self) -> Dict[str, Any]:
        """Return current config as a dict for create_amplifier()."""
        return {
            "host": self._wifi_host.text().strip(),
            "port": self._wifi_port.value(),
            "nch": int(self._wifi_nch.currentText()),
            "srate": float(self._wifi_srate.currentText()),
        }

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply CLI-provided values to the config panel fields."""
        for key, val in overrides.items():
            if not val:
                continue
            if key == "host":
                self._wifi_host.setText(str(val))
            elif key == "port":
                self._wifi_port.setValue(int(val))


# ============================================================================
# Section 8d — Processing Config Dialog
# ============================================================================

class ProcessingConfigDialog(QDialog):
    """实时处理参数设置对话框。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("处理设置")
        self.setMinimumWidth(380)
        self.setStyleSheet("""
            QDialog { background-color: #2d2d35; color: #e0e0e0; }
            QLabel { color: #ddd; font-size: 10px; }
            QRadioButton { color: #ddd; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3a3a42; color: #fff;
                border: 1px solid #666; border-radius: 3px; padding: 3px 6px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #4a9eff;
            }
            QPushButton {
                background-color: #4a4a55; color: white;
                border: 1px solid #666; border-radius: 3px;
                padding: 5px 16px; font-size: 10px;
            }
            QPushButton:hover { background-color: #5a5a66; }
        """)

        layout = QFormLayout(self)

        # Notch frequency
        self._notch_combo = QComboBox()
        self._notch_combo.addItems(["50 Hz", "60 Hz"])
        self._notch_combo.setCurrentText("50 Hz")
        layout.addRow("工频频率:", self._notch_combo)

        # Filter order
        self._order_spin = QSpinBox()
        self._order_spin.setRange(2, 10)
        self._order_spin.setValue(4)
        layout.addRow("滤波器阶数:", self._order_spin)

        # Artifact sigma
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(2.0, 20.0)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setValue(ARTIFACT_SIGMA)
        layout.addRow("伪迹阈值:", self._sigma_spin)

        # Baseline tau
        self._tau_spin = QDoubleSpinBox()
        self._tau_spin.setRange(0.1, 10.0)
        self._tau_spin.setSingleStep(0.5)
        self._tau_spin.setValue(1.0)
        layout.addRow("基线时间常数 (秒):", self._tau_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self) -> Dict[str, Any]:
        return {
            "line_freq": 60.0 if "60" in self._notch_combo.currentText() else 50.0,
            "filter_order": self._order_spin.value(),
            "artifact_sigma": self._sigma_spin.value(),
            "baseline_tau": self._tau_spin.value(),
        }


# ============================================================================
# Section 9 — RealTimeEEGWindow (refactored main window)
# ============================================================================

class RealTimeEEGWindow(QMainWindow):
    """OpenBCI WiFi Shield 实时脑电/肌电/心电可视化主窗口。"""

    def __init__(self, amp_key: str = "wifi_shield", **amp_config) -> None:
        super().__init__()
        self.setWindowTitle("MetaBCI — 实时脑电/肌电/心电 (WiFi Shield)")
        self.resize(1400, 920)

        # ---- Amplifier state ----------------------------------------------
        self._amp_key = amp_key
        self._amp_config = amp_config
        self._adapter: Optional[AbstractAmplifier] = None
        self._connected = False
        self._streaming = False

        # ---- 统一信号处理器 (16 通道) -------------------------------------
        self.processor: Optional[SignalProcessor] = None

        # ---- Ring buffer — 按硬件通道索引直接存取 --------------------------
        self._buffer_secs = DEFAULT_BUFFER_SECS
        self._visible_secs = DEFAULT_VISIBLE_SECS
        self._buffer: Optional[np.ndarray] = None  # (n_hw_channels, buflen)
        self._write_idx: int = 0
        self._filled: int = 0
        self._total_hw_channels: int = MAX_HARDWARE_CHANNELS

        # ---- Wheel event tracking -----------------------------------------
        self._plot_widgets_for_wheel: Dict[int, Any] = {}

        # ---- FPS tracking -------------------------------------------------
        self._frame_times: deque = deque(maxlen=30)

        # ---- Data recording ------------------------------------------------
        self._csv_file: Any = None
        self._csv_path: str = ""
        self._recording_dir = _os.path.join(_this_dir, "recordings")

        # ---- Build UI ------------------------------------------------------
        self._setup_ui()

        # ---- Timer --------------------------------------------------------
        self._timer = QTimer(self)
        self._timer.setInterval(REFRESH_INTERVAL_MS)
        self._timer.timeout.connect(self._update_display)

        # ---- Show config ---------------------------------------------------
        entry = self._find_entry(self._amp_key)
        if entry is not None:
            self._config_panel.show_for_category(entry["cat"], entry)
            self._config_panel.apply_overrides(self._amp_config)
        self._status.showMessage("请在上方配置参数，然后点击 ▶ 开始 进行连接")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # -- Row 1: Amplifier label + config + start -----------------------
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        row1.addWidget(QLabel("OpenBCI WiFi Shield"))
        self._amp_label = QLabel("16通道 / 500Hz")
        self._amp_label.setStyleSheet("color: #4ac0ff; font-weight: bold; font-size: 11px;")
        row1.addWidget(self._amp_label)

        # Config panel
        self._config_panel = AmplifierConfigWidget()
        row1.addWidget(self._config_panel, stretch=1)

        # Start / Stop
        self._start_btn = QPushButton("▶  开始")
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; "
            "font-weight: bold; padding: 4px 14px; }"
        )
        self._start_btn.clicked.connect(self._toggle_stream)
        row1.addWidget(self._start_btn)

        # Processing config gear button
        self._proc_cfg_btn = QToolButton()
        self._proc_cfg_btn.setText("⚙")
        self._proc_cfg_btn.setToolTip("处理设置…")
        self._proc_cfg_btn.setStyleSheet("""
            QToolButton {
                color: #ccc; border: none;
                font-size: 16px; padding: 2px 4px;
            }
            QToolButton:hover { color: #fff; }
        """)
        self._proc_cfg_btn.clicked.connect(self._on_processing_config)
        row1.addWidget(self._proc_cfg_btn)

        # Visible window
        row1.addWidget(QLabel("窗口:"))
        self._window_spin = QSpinBox()
        self._window_spin.setRange(1, 60)
        self._window_spin.setValue(int(self._visible_secs))
        self._window_spin.setSuffix(" s")
        self._window_spin.valueChanged.connect(self._on_window_changed)
        row1.addWidget(self._window_spin)

        # FPS
        self._fps_label = QLabel("帧率: --")
        self._fps_label.setStyleSheet("color: #aaa; font-family: monospace;")
        row1.addWidget(self._fps_label)

        root.addLayout(row1)

        # -- Status bar ----------------------------------------------------
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("就绪 — 配置 WiFi Shield 参数并按开始")

        # -- Plot regions ---------------------------------------------------
        self._splitter = QSplitter(Qt.Vertical)
        self._groups: Dict[str, _ChannelGroup] = {}
        self._plot_widgets: Dict[str, pg.PlotWidget] = {}
        self._group_boxes: Dict[str, QGroupBox] = {}
        root.addWidget(self._splitter, stretch=1)

    # ------------------------------------------------------------------
    # Amplifier lifecycle
    # ------------------------------------------------------------------

    def _do_connect(self) -> bool:
        """Create adapter and connect to the selected amplifier.

        Returns True on success, False on failure (error shown in status bar).
        """
        # If already connected, disconnect first
        if self._connected and self._adapter is not None:
            try:
                self._adapter.disconnect()
            except Exception:
                pass
            self._adapter = None
            self._connected = False

        # Find registry entry
        entry = self._find_entry(self._amp_key)
        if entry is None:
            self._status.showMessage(f"未知放大器: {self._amp_key}")
            return False

        # Check availability
        if not entry.get("available", True):
            self._status.showMessage(
                f"{entry['label']} — 请安装 brainflow (需要 Python >= 3.9)"
            )
            return False

        # Build override config from CLI args + panel config
        overrides = dict(self._amp_config)
        panel_cfg = self._config_panel.get_config()
        overrides.update(panel_cfg)

        # Validate required fields before attempting connection
        if entry.get("needs_ip"):
            ip_addr = overrides.get("ip_address", "").strip()
            if not ip_addr:
                self._status.showMessage(
                    "⚠  请在上方 IP 栏输入设备 IP 地址 "
                    "(例如 OpenBCI WiFi 热点: 192.168.4.1)"
                )
                return False
        if entry.get("key") == "wifi_shield":
            host = overrides.get("host", "").strip()
            if not host:
                self._status.showMessage(
                    "⚠  请输入 WiFi Shield IP 地址 (例如 192.168.4.1)"
                )
                return False

        # Create adapter
        try:
            self._adapter = create_amplifier(entry, **overrides)
        except Exception as exc:
            self._status.showMessage(f"创建放大器错误: {exc}")
            self._adapter = None
            return False

        # Connect
        self._status.showMessage(f"正在连接 {entry['label']}...")
        QApplication.processEvents()  # force status bar update
        try:
            self._adapter.connect()
        except Exception as exc:
            err_msg = str(exc)
            tips = ""
            if "timeout" in err_msg.lower() or "refused" in err_msg.lower():
                tips = (" | 检查: 1) 设备是否开机? "
                        "2) IP 是否正确? 3) 防火墙是否阻止?")
            elif "host" in err_msg.lower() or "resolve" in err_msg.lower():
                tips = " | 检查: IP 地址是否有效?"
            self._status.showMessage(f"连接错误: {err_msg}{tips}")
            self._adapter = None
            return False

        self._connected = True

        # Post-connect: discover channels, build layout
        self._setup_channels()

        # Update amp label
        nch = self._adapter.channel_count
        srate = self._adapter.sample_rate
        self._amp_label.setText(f"{nch}通道 / {srate:.0f}Hz")

        self._status.showMessage(
            f"已连接 — {srate:.0f} Hz, {nch} 通道"
        )
        return True

    def _setup_channels(self) -> None:
        """连接后初始化：重建绘图区域 + 分配缓冲区 + 创建处理器。"""
        if self._adapter is None:
            return
        self._rebuild_plot_groups()
        self._allocate_buffer()
        self._create_processor()

    def _find_entry(self, key: str) -> Optional[Dict[str, Any]]:
        for entry in AMPLIFIER_REGISTRY:
            if entry["key"] == key:
                return entry
        return None

    # ------------------------------------------------------------------
    # Plot groups — 按硬件通道索引建组
    # ------------------------------------------------------------------

    def _rebuild_plot_groups(self) -> None:
        while self._splitter.count() > 0:
            w = self._splitter.widget(0)
            w.hide()
            w.setParent(None)
            w.deleteLater()
        self._groups.clear()
        self._plot_widgets.clear()
        self._group_boxes.clear()

        # 每个组独立的通道名称和颜色
        all_names = ALL_CHANNEL_NAMES
        eeg_names = [all_names[i] if i < len(all_names) else f"CH{i+1}"
                     for i in EEG_CHANNELS]
        emg_names = [all_names[i] if i < len(all_names) else f"CH{i+1}"
                     for i in EMG_CHANNELS]
        ecg_names = [all_names[i] if i < len(all_names) else f"CH{i+1}"
                     for i in ECG_CHANNELS]
        eeg_colors = [TRACE_COLORS[i % len(TRACE_COLORS)] for i in range(len(EEG_CHANNELS))]
        emg_colors = [TRACE_COLORS[(len(EEG_CHANNELS) + i) % len(TRACE_COLORS)]
                      for i in range(len(EMG_CHANNELS))]
        ecg_colors = [TRACE_COLORS[(len(EEG_CHANNELS) + len(EMG_CHANNELS) + i)
                                   % len(TRACE_COLORS)]
                      for i in range(len(ECG_CHANNELS))]

        for group_name in GROUP_PRIORITY:
            ch_list = REGION_CFG[group_name]["channels"]
            if not ch_list:
                continue
            cfg = REGION_CFG[group_name]

            box = QGroupBox()
            box.setFont(QFont("Segoe UI", 10, QFont.Bold))
            layout = QVBoxLayout(box)
            layout.setContentsMargins(2, 2, 2, 2)

            pw = pg.PlotWidget()
            pw.getPlotItem().getViewBox().setBackgroundColor((18, 18, 22))
            layout.addWidget(pw)

            if group_name == "EEG":
                cg = _ChannelGroup(
                    plot_item=pw.getPlotItem(),
                    channel_idxs=EEG_CHANNELS,
                    channel_names=eeg_names,
                    colors=eeg_colors,
                    trace_spacing_uv=cfg["trace_spacing_uv"],
                    display_target_uv=cfg["display_target_uv"],
                    gain_multiplier=cfg["gain_multiplier"],
                )
            elif group_name == "EMG":
                cg = _ChannelGroup(
                    plot_item=pw.getPlotItem(),
                    channel_idxs=EMG_CHANNELS,
                    channel_names=emg_names,
                    colors=emg_colors,
                    trace_spacing_uv=cfg["trace_spacing_uv"],
                    display_target_uv=cfg["display_target_uv"],
                    gain_multiplier=cfg["gain_multiplier"],
                )
            else:  # ECG
                cg = _ChannelGroup(
                    plot_item=pw.getPlotItem(),
                    channel_idxs=ECG_CHANNELS,
                    channel_names=ecg_names,
                    colors=ecg_colors,
                    trace_spacing_uv=cfg["trace_spacing_uv"],
                    display_target_uv=cfg["display_target_uv"],
                    gain_multiplier=cfg["gain_multiplier"],
                )

            self._groups[group_name] = cg
            self._plot_widgets[group_name] = pw

            names_str = ", ".join((eeg_names if group_name == "EEG"
                                   else emg_names if group_name == "EMG"
                                   else ecg_names)[:8])
            if len(ch_list) > 8:
                names_str += ", …"
            box.setTitle(
                f"{group_name} ({len(ch_list)}ch)  "
                f"1-30 Hz 陷波 50 Hz  |  {names_str}"
            )
            self._splitter.addWidget(box)
            self._group_boxes[group_name] = box

        n_visible = len(self._groups)
        if n_visible == 3:
            self._splitter.setSizes([380, 280, 160])
        elif n_visible == 2:
            self._splitter.setSizes([500, 300])
        elif n_visible == 1:
            self._splitter.setSizes([800])

        for pw in self._plot_widgets.values():
            # 用 eventFilter 拦截滚轮（pyqtgraph ViewBox 内部会吞掉 wheelEvent）
            pw.viewport().installEventFilter(self)
            self._plot_widgets_for_wheel[id(pw)] = pw

    def _allocate_buffer(self) -> None:
        fs = self._adapter.sample_rate if self._adapter else DEFAULT_SAMPLE_RATE
        buflen = int(fs * self._buffer_secs)
        self._buffer = np.full(
            (self._total_hw_channels, buflen), np.nan, dtype=np.float32)
        self._write_idx = 0
        self._filled = 0

    def _create_processor(self) -> None:
        fs = self._adapter.sample_rate if self._adapter else DEFAULT_SAMPLE_RATE
        self.processor = SignalProcessor(
            channels=self._total_hw_channels,
            sample_rate=fs,
            low_hz=1.0, high_hz=30.0,
            notch_hz=50.0, baseline_seconds=1.0,
        )

    # ------------------------------------------------------------------
    # Stream control
    # ------------------------------------------------------------------

    def _toggle_stream(self) -> None:
        if self._streaming:
            self._stop_stream()
        else:
            self._start_stream()

    def _start_stream(self) -> None:
        # If not yet connected, connect first
        if not self._connected:
            success = self._do_connect()
            if not success:
                return

        if self._adapter is None:
            self._status.showMessage("未连接放大器。")
            return
        try:
            self._adapter.start_acquisition()

            # -- 自动创建 CSV 记录文件 -----------------------------------------
            _os.makedirs(self._recording_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._csv_path = _os.path.join(
                self._recording_dir, f"eeg_{ts}.csv")
            self._csv_file = open(self._csv_path, "w", encoding="utf-8")
            # 写入表头: timestamp, channel_names...
            ch_names = self._adapter.channel_names
            self._csv_file.write("timestamp," + ",".join(ch_names) + "\n")
            self._csv_file.flush()
            self._sample_count_since_open = 0

            self._streaming = True
            self._start_btn.setText("⏸  停止")
            self._start_btn.setStyleSheet(
                "QPushButton { background-color: #c0392b; color: white; "
                "font-weight: bold; padding: 4px 14px; }"
            )
            self._timer.start()
            self._status.showMessage(
                f"采集中… {self._adapter.sample_rate:.0f}Hz, "
                f"{self._adapter.channel_count}通道 | 记录: {self._csv_path}"
            )
        except Exception as exc:
            self._status.showMessage(f"启动错误: {exc}")

    def _stop_stream(self) -> None:
        self._timer.stop()
        # -- 关闭 CSV 记录文件 -------------------------------------------------
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
            saved_path = self._csv_path
            self._csv_file = None
            self._csv_path = ""
        else:
            saved_path = ""
        if self._adapter is not None:
            try:
                self._adapter.stop_acquisition()
            except Exception:
                pass
        self._streaming = False
        self._start_btn.setText("▶  开始")
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; "
            "font-weight: bold; padding: 4px 14px; }"
        )
        msg = "采集已停止。(仍保持连接 — 点击开始恢复)"
        if saved_path:
            msg += f" | 已保存: {saved_path}"
        self._status.showMessage(msg)

    # ------------------------------------------------------------------
    # Display update (timer callback)
    # ------------------------------------------------------------------

    def _update_display(self) -> None:
        if self._adapter is None or not self._streaming:
            return
        if self._buffer is None or self.processor is None:
            return

        t0 = time.perf_counter()

        try:
            raw = self._adapter.get_data()
        except Exception as exc:
            self._status.showMessage(f"读取错误: {exc}")
            return

        if raw.size == 0 or raw.shape[1] == 0:
            self._update_fps(t0)
            return

        n_new = raw.shape[1]
        data = raw.astype(np.float64)  # (n_channels, n_new)

        # ---- 统一滤波 (参照 eeg_display.py 的 update_data) ------------------
        # 处理器要求 (n_samples, n_channels)，输入 (n_channels, n_new)
        if data.shape[0] > self._total_hw_channels:
            data = data[:self._total_hw_channels, :]
        filtered = self.processor.process(data)  # → (n_channels, n_new)

        # EEG 通道做 CAR
        eeg_data = filtered[EEG_CHANNELS, :]
        if len(EEG_CHANNELS) > 1:
            car_mean = np.nanmean(eeg_data, axis=0, keepdims=True)
            filtered[EEG_CHANNELS, :] = eeg_data - car_mean

        # ---- 写入环形缓冲区 (按硬件通道索引) --------------------------------
        capacity = self._buffer.shape[1]
        if n_new >= capacity:
            self._buffer[:, :] = filtered[:, -capacity:]
            self._write_idx = 0
            self._filled = capacity
        else:
            start = self._write_idx
            end = start + n_new
            if end <= capacity:
                self._buffer[:, start:end] = filtered
            else:
                first = capacity - start
                self._buffer[:, start:] = filtered[:, :first]
                self._buffer[:, :end % capacity] = filtered[:, first:]
            self._write_idx = end % capacity
            self._filled = min(self._filled + n_new, capacity)

        # ---- CSV 记录 -------------------------------------------------------
        if self._csv_file is not None:
            fs = self._adapter.sample_rate
            total_samples = (self._filled if self._filled < capacity
                             else self._write_idx + capacity) if self._filled else 0
            base_sample = total_samples - n_new
            lines = []
            for j in range(n_new):
                t = (base_sample + j) / fs
                vals = ",".join(f"{filtered[ch, j]:.6f}"
                                for ch in range(filtered.shape[0]))
                lines.append(f"{t:.6f},{vals}\n")
            self._csv_file.writelines(lines)
            self._csv_file.flush()

        # ---- 渲染各区域 -----------------------------------------------------
        ordered = self._ordered_buffer()
        if ordered.shape[1] == 0:
            self._update_fps(t0)
            return

        for group_name, cg in self._groups.items():
            cg.update_plot(ordered, self._adapter.sample_rate,
                           self._visible_secs)

        self._update_fps(t0)

    def _ordered_buffer(self) -> np.ndarray:
        """按时间顺序返回缓冲区数据 (参照 eeg_display.py)。"""
        if self._buffer is None or self._filled == 0:
            return np.empty((self._total_hw_channels, 0), dtype=np.float32)
        if self._filled < self._buffer.shape[1]:
            return self._buffer[:, :self._filled]
        idx = self._write_idx
        return np.concatenate(
            (self._buffer[:, idx:], self._buffer[:, :idx]), axis=1)

    def _update_fps(self, t0: float) -> None:
        self._frame_times.append(time.perf_counter())
        if len(self._frame_times) >= 2:
            dt = self._frame_times[-1] - self._frame_times[0]
            if dt > 0:
                fps = (len(self._frame_times) - 1) / dt
                self._fps_label.setText(
                    f"帧率: {fps:4.1f} | 缓冲: {self._filled}")

    # ------------------------------------------------------------------
    # Worker panel
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # UI interactions
    # ------------------------------------------------------------------

    def _on_processing_config(self) -> None:
        """打开处理配置对话框。"""
        dlg = ProcessingConfigDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        vals = dlg.get_values()

        # Apply to unified processor
        if self.processor is not None:
            self.processor.set_line_freq(vals["line_freq"])
            self.processor.set_filter_order(vals["filter_order"])
            self.processor.set_baseline_tau(vals["baseline_tau"])

        self._status.showMessage("处理设置已应用")

    def _on_window_changed(self, value: int) -> None:
        self._visible_secs = float(value)

    def eventFilter(self, obj, event):
        """截获 PlotWidget viewport 上的滚轮事件。"""
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Wheel:
            for pw in getattr(self, '_plot_widgets_for_wheel', {}).values():
                if obj is pw.viewport():
                    self._on_wheel(event, pw)
                    return True
        return super().eventFilter(obj, event)

    def _on_wheel(self, event, plot_widget: pg.PlotWidget) -> None:
        delta = event.angleDelta().y()
        modifiers = event.modifiers()

        if modifiers & Qt.ControlModifier:
            # Ctrl + scroll: 水平缩放 — 改变时间窗口
            if delta > 0:
                self._visible_secs = min(60.0, self._visible_secs + 0.5)
            else:
                self._visible_secs = max(1.0, self._visible_secs - 0.5)
            self._window_spin.blockSignals(True)
            self._window_spin.setValue(int(self._visible_secs))
            self._window_spin.blockSignals(False)
        else:
            # 普通滚轮: 垂直缩放 — 改变波形幅度增益
            factor = 1.15 if delta > 0 else 0.87  # ~15% per step
            for cg in self._groups.values():
                cg.set_gain(cg._gain_mul * factor)

    # ------------------------------------------------------------------
    # Clean shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self._stop_stream()
        # 确保 CSV 文件关闭
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
        if self._adapter is not None:
            try:
                self._adapter.disconnect()
            except Exception:
                pass
        super().closeEvent(event)


# ============================================================================
# Section 10 — main()
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time EEG/EMG/ECG viewer — unified amplifier + worker GUI",
    )
    parser.add_argument(
        "--amp", dest="amp_key",
        choices=[e["key"] for e in AMPLIFIER_REGISTRY],
        default="wifi_shield",
        help="Amplifier type (default: cyton_daisy_wifi)",
    )
    # BrainFlow-specific
    parser.add_argument("--serial-port", default="")
    parser.add_argument("--ip-address", default="")
    parser.add_argument("--ip-port", type=int, default=0)
    # TCP-specific
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--srate", type=float, default=0)
    parser.add_argument("--nch", type=int, default=0)
    args = parser.parse_args()

    amp_config: Dict[str, Any] = {}
    if args.serial_port:
        amp_config["serial_port"] = args.serial_port
    if args.ip_address:
        amp_config["ip_address"] = args.ip_address
    if args.ip_port:
        amp_config["ip_port"] = args.ip_port
    if args.host:
        amp_config["host"] = args.host
    if args.port:
        amp_config["port"] = args.port
    if args.srate:
        amp_config["srate"] = args.srate
    if args.nch:
        amp_config["nch"] = args.nch

    # Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("MetaBCI-EEG")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(35, 35, 38))
    palette.setColor(QPalette.WindowText, QColor(235, 235, 235))
    palette.setColor(QPalette.Base, QColor(28, 28, 30))
    palette.setColor(QPalette.Text, QColor(235, 235, 235))
    palette.setColor(QPalette.Button, QColor(50, 50, 55))
    palette.setColor(QPalette.ButtonText, QColor(235, 235, 235))
    palette.setColor(QPalette.Highlight, QColor(0, 122, 204))
    app.setPalette(palette)
    app.setFont(QFont("Segoe UI", 9))

    # Global stylesheet for dark-theme form controls
    app.setStyleSheet("""
        QLineEdit {
            background-color: #3a3a42;
            color: #ffffff;
            border: 1px solid #666;
            border-radius: 3px;
            padding: 2px 6px;
        }
        QLineEdit:focus { border: 1px solid #4a9eff; }
        QSpinBox {
            background-color: #3a3a42;
            color: #ffffff;
            border: 1px solid #666;
            border-radius: 3px;
            padding: 2px 6px;
        }
        QSpinBox:focus { border: 1px solid #4a9eff; }
        QSpinBox::up-button, QSpinBox::down-button {
            background-color: #4a4a55;
            border: none;
            width: 14px;
        }
        QSpinBox::up-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-bottom: 5px solid #ccc; }
        QSpinBox::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #ccc; }
        QGroupBox {
            color: #ddd;
            font-weight: bold;
            border: 1px solid #555;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            padding: 0 8px;
        }
        QStatusBar {
            color: #ccc;
            background-color: #2a2a30;
        }
    """)

    window = RealTimeEEGWindow(amp_key=args.amp_key, **amp_config)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
