# -*- coding: utf-8 -*-
"""实时 EEG/EMG/ECG 多区波形显示 + 标签标注控件。

属于 metabci.brainflow.gui 包。

信号处理增强 (from Fork B real_time_eeg.py)：
- SOS 带通(1-30Hz) + 陷波(50/60Hz) 组合滤波
- 共同平均参考 (CAR) — 从 EEG 通道减去均值
- 运行时重配置：set_line_freq / set_filter_order / set_baseline_tau
- 线性硬裁剪 (np.clip) 替代 tanh 软裁剪，保持波形不畸变
- setXRange 每帧固定，滚动稳定

使用 metabci.brainflow.logger 记录日志；滤波器设计对齐
metabci.brainda.algorithms.decomposition.base.FilterBank。
"""

from __future__ import annotations
from typing import List, Optional

import time

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QFont
from PySide6.QtWidgets import (
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QVBoxLayout, QWidget,
)

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False
    pg = None

try:
    from scipy.signal import butter, sosfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# MetaBCI logger
try:
    from ..logger import get_logger
except ImportError:
    from metabci.brainflow.logger import get_logger  # type: ignore[no-redef]

logger = get_logger("eeg_display")


# ============================================================
# 通道配置 — 与实际硬件接线一一对应
# ============================================================
ALL_CHANNEL_NAMES = [
    "FP1",  # CH1  EEG
    "FP2",  # CH2  EEG
    "C3",   # CH3  EEG
    "C4",   # CH4  EEG
    "P7",   # CH5  EEG
    "P8",   # CH6  EEG
    "EMG1", # CH7  咽喉肌电
    "EMG2", # CH8  咽喉肌电
    "EMG3", # CH9  咽喉肌电
    "EMG4", # CH10 咽喉肌电
    "F3",   # CH11 EEG
    "F4",   # CH12 EEG
    "ECG",  # CH13 锁骨心电
    "Cz",   # CH14 EEG
    "EMG5", # CH15 胸部肌电
    "EMG6", # CH16 胸部肌电
]

EEG_CHANNELS = [0, 1, 2, 3, 4, 5, 10, 11, 13]  # FP1,FP2,C3,C4,P7,P8,F3,F4,Cz
EMG_CHANNELS = [6, 7, 8, 9, 14, 15]             # EMG1-4, EMG5-6
ECG_CHANNELS = [12]                               # ECG

TRACE_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    "#5f9ea0", "#ff6347", "#6a5acd", "#20b2aa", "#4682b4",
    "#cd5c5c", "#db7093",
]
EEG_COLOR = "#4CAF50"
EMG_COLOR = "#FF9800"
ECG_COLOR = "#F44336"


# ============================================================
# 信号处理器 — 去基线 + 带通滤波（复刻 stroke RealtimeEEGProcessor）
# ============================================================

class SignalProcessor:
    """去基线 + 带通滤波 + 陷波 + CAR (Fork B 增强版)。

    滤波器设计使用与 metabci.brainda.algorithms.decomposition.base.FilterBank
    相同的 scipy.signal.butter / sosfilt 工具链。

    新增 (vs Fork A)：
    - CAR 共同平均参考
    - 运行时重配置 set_line_freq / set_filter_order / set_baseline_tau
    - reset() 清除滤波器状态
    """

    def __init__(
        self,
        channels: int,
        sample_rate: float,
        low_hz: float = 1.0,
        high_hz: float = 30.0,
        notch_hz: float = 50.0,
        baseline_seconds: float = 1.0,
    ):
        self.channels = int(channels)
        self.sample_rate = float(sample_rate)
        self._notch_hz = float(notch_hz)
        self._low_hz = float(low_hz)
        self._high_hz = float(high_hz)

        self._all_sos: List[np.ndarray] = []
        self._rebuild_filters()

        self.baseline = np.zeros((self.channels,), dtype=np.float64)
        self.baseline_ready = np.zeros((self.channels,), dtype=bool)
        self.baseline_alpha = 1.0 / max(1.0, sample_rate * baseline_seconds)

        logger.info(
            "SignalProcessor: %dch @ %.1fHz, bandpass %.1f-%.1fHz, notch %.1fHz",
            channels, sample_rate, low_hz, high_hz, notch_hz)

    # -- runtime reconfiguration (Fork B) -------------------------------

    def set_line_freq(self, freq: float) -> None:
        """切换工频频率 50↔60Hz。"""
        self._notch_hz = float(freq)
        self._rebuild_filters()
        logger.debug("Notch → %.1f Hz", freq)

    def set_filter_order(self, order: int) -> None:
        """切换滤波器阶数。"""
        self._rebuild_filters(order=max(2, min(10, int(order))))

    def set_baseline_tau(self, seconds: float) -> None:
        """改变基线时间常数。"""
        self.baseline_alpha = 1.0 / max(1.0, self.sample_rate * max(0.1, seconds))

    def reset(self) -> None:
        """清除滤波器状态（放大器重启后调用）。"""
        self.baseline = np.zeros((self.channels,), dtype=np.float64)
        self.baseline_ready = np.zeros((self.channels,), dtype=bool)
        if self.sos is not None:
            self.zi = np.zeros((self.sos.shape[0], 2, self.channels), dtype=np.float64)

    def _rebuild_filters(self, order: int = 4) -> None:
        self._all_sos.clear()
        if not HAS_SCIPY:
            self.sos = None
            self.zi = None
            return
        nyquist = self.sample_rate / 2.0
        high = min(self._high_hz, nyquist * 0.90)
        if self._low_hz > 0 and high > self._low_hz:
            self._all_sos.append(butter(
                order, [self._low_hz, high], btype="bandpass",
                fs=self.sample_rate, output="sos"))
        if self._notch_hz > 0:
            w0 = self._notch_hz / nyquist
            Q = 30.0
            self._all_sos.append(butter(
                2, [w0 - w0 / Q, w0 + w0 / Q], btype="bandstop",
                output="sos"))
        if self._all_sos:
            combined = self._all_sos[0]
            for s in self._all_sos[1:]:
                combined = np.vstack([combined, s])
            self.sos = combined
        else:
            self.sos = None
        self.zi = np.zeros(
            (self.sos.shape[0], 2, self.channels), dtype=np.float64
        ) if self.sos is not None else None

    # -- processing -----------------------------------------------------

    def process_batch(self, rows_uv, eeg_indices: list | None = None
                      ) -> np.ndarray:
        """去基线 + SOS 滤波 + 可选 CAR。

        Parameters
        ----------
        rows_uv : (n_samples, n_channels)
        eeg_indices : list of int or None
            EEG 通道列索引，提供时启用 CAR。
        """
        src = np.asarray(rows_uv, dtype=np.float64)
        if src.ndim == 1:
            src = src.reshape(1, -1)
        n_samples = src.shape[0]
        n_ch = min(src.shape[1], self.channels)

        values = np.full((n_samples, self.channels), np.nan, dtype=np.float64)
        if n_ch:
            values[:, :n_ch] = src[:, :n_ch]

        detrended = np.zeros((n_samples, self.channels), dtype=np.float64)
        valid_mask = np.isfinite(values)

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

        if self.sos is not None and self.zi is not None:
            zic = self.zi[:, :, :n_ch] if self.zi.shape[2] > n_ch else self.zi
            filtered, zic = sosfilt(self.sos, detrended[:, :n_ch], axis=0, zi=zic)
            detrended[:, :n_ch] = filtered
            if self.zi.shape[2] > n_ch:
                self.zi[:, :, :n_ch] = zic

        filtered = detrended

        # CAR (Fork B) — mean of EEG channels as common-mode reference
        if eeg_indices and len(eeg_indices) > 1:
            eeg_cols = [c for c in eeg_indices if c < self.channels]
            if len(eeg_cols) > 1:
                car_mean = np.nanmean(filtered[:, eeg_cols], axis=1, keepdims=True)
                filtered[:, eeg_cols] -= car_mean

        filtered[~valid_mask] = np.nan
        return filtered.astype(np.float32, copy=False)


# ============================================================
# 单区通道组 — 复刻 stroke LiveEEGWindow 的偏移堆叠显示
# ============================================================

class _ChannelGroup:
    """一组通道在一个 PlotWidget 内用垂直偏移堆叠显示。"""

    def __init__(
        self,
        plot_item: pg.PlotItem,
        channel_idxs: list[int],
        channel_names: list[str],
        colors: list[str],
        trace_spacing_uv: float = 160.0,
        display_target_uv: float = 42.0,
        display_soft_limit_uv: float = 72.0,
        display_min_gain: float = 0.02,
        display_max_gain: float = 5.0,
        gain_multiplier: float = 1.0,
        max_draw_points: int = 1200,
    ):
        self.channel_idxs = list(channel_idxs)
        self.n_ch = len(channel_idxs)
        self._spacing = trace_spacing_uv
        self._target = display_target_uv
        self._soft = display_soft_limit_uv
        self._min_gain = display_min_gain * gain_multiplier
        self._max_gain = display_max_gain * gain_multiplier
        self._gain_mul = gain_multiplier
        self._max_pts = max_draw_points

        # 垂直偏移：最上面通道偏移最小
        self._offsets = np.asarray([
            (self.n_ch - 1 - i) * trace_spacing_uv
            for i in range(self.n_ch)
        ], dtype=np.float32)

        # Dark theme (Fork B) + axis styling
        plot_item.clear()
        plot_item.showGrid(x=True, y=True, alpha=0.15)
        plot_item.setMouseEnabled(x=True, y=False)
        plot_item.hideButtons()
        plot_item.setMenuEnabled(False)
        plot_item.getViewBox().setBackgroundColor((18, 18, 22))
        plot_item.getAxis("bottom").setPen(pg.mkPen(color="#777"))
        plot_item.getAxis("bottom").setTextPen(pg.mkPen(color="#aaa"))
        plot_item.getAxis("bottom").setTickFont(QFont("Segoe UI", 9))

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

        plot_item.setYRange(
            -trace_spacing_uv,
            float(self._offsets[0] + trace_spacing_uv),
            padding=0.02,
        )

        self._plot_item = plot_item
        self._curves: List[pg.PlotDataItem] = []
        for i in range(self.n_ch):
            curve = plot_item.plot(
                pen=pg.mkPen(colors[i % len(colors)], width=1.1),
                connect="finite", antialias=True)
            curve.setClipToView(True)
            curve.setDownsampling(auto=False)
            self._curves.append(curve)

    def update_plot(self, ordered_data: np.ndarray, fs: float,
                    window_s: float):
        """更新曲线。

        Parameters
        ----------
        ordered_data : ndarray (n_all_channels, n_samples)
        fs : 采样率
        window_s : 显示窗口秒数
        """
        n = ordered_data.shape[1]
        if n == 0:
            return

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

        for i, (ch_idx, curve) in enumerate(
            zip(self.channel_idxs, self._curves)
        ):
            if ch_idx >= ordered_data.shape[0]:
                continue
            y = self._scale_trace(ordered_data[ch_idx])
            curve.setData(x, y + self._offsets[i], connect="finite")

    def _scale_trace(self, values_uv: np.ndarray) -> np.ndarray:
        """自适应增益 + soft clipping。"""
        y = np.asarray(values_uv, dtype=np.float32).copy()
        finite = np.isfinite(y)
        if not np.any(finite):
            return y

        yf = y[finite].astype(np.float32, copy=False)
        center = float(np.median(yf))
        dev = np.abs(yf - center)
        amp95 = float(np.percentile(dev, 95))
        if np.isfinite(amp95) and amp95 > 1e-6:
            gain = self._target / amp95
            gain = float(np.clip(gain, self._min_gain, self._max_gain))
        else:
            gain = 1.0
        yf = (yf - center) * gain
        yf = np.clip(yf, -self._soft, self._soft)
        y[finite] = yf.astype(np.float32, copy=False)
        return y


# ============================================================
# MultiRegionEEGWidget — EEG / EMG / ECG 三区显示
# ============================================================

class MultiRegionEEGWidget(QWidget):
    """EEG / EMG / ECG 三区实时波形。

    只显示已分配到组的通道，空闲通道自动隐藏=无噪声。
    每区独立偏移堆叠，复刻 stroke 项目的显示方式。
    """

    def __init__(
        self,
        eeg_channels: Optional[list[int]] = None,
        emg_channels: Optional[list[int]] = None,
        ecg_channels: Optional[list[int]] = None,
        srate: float = 500.0,
        visible_duration: float = 8.0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        if not HAS_PYQTGRAPH:
            raise ImportError("pyqtgraph 未安装。pip install pyqtgraph")

        self.srate = srate
        self.visible_duration = visible_duration

        self.eeg_channels = EEG_CHANNELS if eeg_channels is None else list(eeg_channels)
        self.emg_channels = EMG_CHANNELS if emg_channels is None else list(emg_channels)
        self.ecg_channels = ECG_CHANNELS if ecg_channels is None else list(ecg_channels)

        # 所有活跃通道的并集（用于缓冲区和处理器）
        self._all_active = sorted(set(
            list(self.eeg_channels) + list(self.emg_channels)
            + list(self.ecg_channels)
        ))
        self.n_total = len(self._all_active)
        # 最大 16 通道处理器（要保持索引正确）
        self.n_processor = max(self._all_active) + 1 if self._all_active else 16

        # 处理器 — EEG: 1-30Hz带通+50Hz陷波
        self.processor = SignalProcessor(
            channels=self.n_processor,
            sample_rate=srate,
            low_hz=1.0,
            high_hz=30.0,
            notch_hz=50.0,
        )

        # 缓冲区
        buf_len = int(srate * visible_duration)
        self._buffer = np.full(
            (self.n_processor, buf_len), np.nan, dtype=np.float32
        )
        self._write_idx = 0
        self._filled = 0

        # 每通道独立颜色（从 TRACE_COLORS 循环取色）
        self._eeg_colors = [
            TRACE_COLORS[i % len(TRACE_COLORS)]
            for i in range(len(self.eeg_channels))
        ]
        self._emg_colors = [
            TRACE_COLORS[(len(self.eeg_channels) + i) % len(TRACE_COLORS)]
            for i in range(len(self.emg_channels))
        ]
        self._ecg_colors = [
            TRACE_COLORS[(len(self.eeg_channels) + len(self.emg_channels) + i)
                         % len(TRACE_COLORS)]
            for i in range(len(self.ecg_channels))
        ]

        # 通道名列表
        self._eeg_names = [
            ALL_CHANNEL_NAMES[i] if i < len(ALL_CHANNEL_NAMES)
            else f"CH{i + 1}"
            for i in self.eeg_channels
        ]
        self._emg_names = [
            ALL_CHANNEL_NAMES[i] if i < len(ALL_CHANNEL_NAMES)
            else f"CH{i + 1}"
            for i in self.emg_channels
        ]
        self._ecg_names = [
            ALL_CHANNEL_NAMES[i] if i < len(ALL_CHANNEL_NAMES)
            else f"CH{i + 1}"
            for i in self.ecg_channels
        ]

        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(4)

        # --- EEG 区 ---
        if self.eeg_channels:
            eeg_grp = QGroupBox(
                f"EEG ({len(self.eeg_channels)}ch: "
                + ", ".join(self._eeg_names) + ")"
            )
            eeg_layout = QVBoxLayout(eeg_grp)
            eeg_layout.setContentsMargins(2, 2, 2, 2)
            self._eeg_glw = pg.GraphicsLayoutWidget()
            self._eeg_glw.setBackground("w")
            eeg_layout.addWidget(self._eeg_glw)
            root.addWidget(eeg_grp, stretch=len(self.eeg_channels))
            self._eeg_group = _ChannelGroup(
                plot_item=self._eeg_glw.addPlot(row=0, col=0),
                channel_idxs=self.eeg_channels,
                channel_names=self._eeg_names,
                colors=self._eeg_colors,
                trace_spacing_uv=280.0,
                display_target_uv=50.0,
            )
        else:
            self._eeg_group = None

        # --- EMG 区 ---
        if self.emg_channels:
            emg_grp = QGroupBox(
                f"EMG ({len(self.emg_channels)}ch: "
                + ", ".join(self._emg_names) + ")"
            )
            emg_layout = QVBoxLayout(emg_grp)
            emg_layout.setContentsMargins(2, 2, 2, 2)
            self._emg_glw = pg.GraphicsLayoutWidget()
            self._emg_glw.setBackground("w")
            emg_layout.addWidget(self._emg_glw)
            root.addWidget(emg_grp, stretch=len(self.emg_channels))
            self._emg_group = _ChannelGroup(
                plot_item=self._emg_glw.addPlot(row=0, col=0),
                channel_idxs=self.emg_channels,
                channel_names=self._emg_names,
                colors=self._emg_colors,
                trace_spacing_uv=220.0,
                gain_multiplier=0.25,
            )
        else:
            self._emg_group = None

        # --- ECG 区 ---
        if self.ecg_channels:
            ecg_grp = QGroupBox(
                f"ECG ({len(self.ecg_channels)}ch: "
                + ", ".join(self._ecg_names) + ")"
            )
            ecg_layout = QVBoxLayout(ecg_grp)
            ecg_layout.setContentsMargins(2, 2, 2, 2)
            self._ecg_glw = pg.GraphicsLayoutWidget()
            self._ecg_glw.setBackground("w")
            ecg_layout.addWidget(self._ecg_glw)
            root.addWidget(ecg_grp, stretch=max(1, len(self.ecg_channels)))
            self._ecg_group = _ChannelGroup(
                plot_item=self._ecg_glw.addPlot(row=0, col=0),
                channel_idxs=self.ecg_channels,
                channel_names=self._ecg_names,
                colors=self._ecg_colors,
                trace_spacing_uv=260.0,
                gain_multiplier=0.25,
            )
        else:
            self._ecg_group = None

    # -----------------------------------------------------------------
    # 数据更新（复刻 stroke _on_timer + _append_samples + _update_plots）
    # -----------------------------------------------------------------

    def update_data(self, data: np.ndarray):
        """接收原始数据块，滤波后更新波形。

        Parameters
        ----------
        data : ndarray (n_channels, n_samples), uV 值
        """
        if data is None or data.size == 0:
            return
        n_new = data.shape[1]
        if n_new == 0:
            return

        # 转置为 (n_samples, n_channels) 供处理器
        samples = data.T  # (n_new, n_channels)
        if samples.shape[1] > self.n_processor:
            samples = samples[:, :self.n_processor]

        filtered = self.processor.process_batch(samples)

        # 写入环形缓冲
        capacity = self._buffer.shape[1]
        rows_to_write = filtered[:, :self.n_processor].T  # (n_ch, n_new)

        if n_new >= capacity:
            self._buffer[:, :] = rows_to_write[:, -capacity:]
            self._write_idx = 0
            self._filled = capacity
        else:
            start = self._write_idx
            end = start + n_new
            if end <= capacity:
                self._buffer[:, start:end] = rows_to_write
            else:
                first = capacity - start
                self._buffer[:, start:] = rows_to_write[:, :first]
                self._buffer[:, :end % capacity] = rows_to_write[:, first:]
            self._write_idx = end % capacity
            self._filled = min(self._filled + n_new, capacity)

        # 更新各区曲线
        ordered = self._ordered_buffer()
        if ordered.shape[1] == 0:
            return

        for group in [self._eeg_group, self._emg_group, self._ecg_group]:
            if group is not None:
                group.update_plot(
                    ordered, self.srate, self.visible_duration,
                )

    def _ordered_buffer(self) -> np.ndarray:
        if self._filled == 0:
            return self._buffer[:, :0]
        if self._filled < self._buffer.shape[1]:
            return self._buffer[:, :self._filled]
        idx = self._write_idx
        return np.concatenate(
            (self._buffer[:, idx:], self._buffer[:, :idx]), axis=1
        )

    # -----------------------------------------------------------------
    # 配置
    # -----------------------------------------------------------------

    def set_visible_duration(self, seconds: float):
        seconds = max(1.0, min(60.0, seconds))
        self.visible_duration = seconds
        new_len = int(self.srate * seconds)
        if new_len > self._buffer.shape[1]:
            old = self._buffer
            self._buffer = np.full(
                (self.n_processor, new_len), np.nan, dtype=np.float32
            )
            self._buffer[:, -old.shape[1]:] = old
        else:
            self._buffer = self._buffer[:, -new_len:]
        self._write_idx = 0
        self._filled = min(self._filled, new_len)
        # 更新各区 X 轴范围
        for group in [self._eeg_group, self._emg_group, self._ecg_group]:
            if group is not None:
                group._plot_item.setXRange(0.0, seconds, padding=0.0)

    def wheelEvent(self, event):
        """鼠标滚轮调整显示窗口大小。"""
        delta = event.angleDelta().y()
        step = 1.0 if abs(delta) < 120 else 2.0
        if delta > 0:
            self.set_visible_duration(self.visible_duration + step)
        else:
            self.set_visible_duration(self.visible_duration - step)
        event.accept()

    def clear(self):
        self._buffer.fill(np.nan)
        self._write_idx = 0
        self._filled = 0


# ============================================================
# LabelPanel — 手动标签标注面板
# ============================================================

DEFAULT_LABELS = [
    ("静息",          1, "1", "#4CAF50"),
    ("想象吞咽",      2, "2", "#2196F3"),
    ("实际吞咽",      3, "3", "#FF9800"),
    ("眨眼",          4, "4", "#9C27B0"),
    ("咬牙",          5, "5", "#F44336"),
    ("自定义1",       6, "6", "#607D8B"),
    ("自定义2",       7, "7", "#795548"),
    ("自定义3",       8, "8", "#00BCD4"),
]


class LabelPanel(QWidget):
    """手动事件标签面板。

    可配置标签按钮 + 键盘快捷键。
    点击按钮时发射 label_triggered 信号。
    """

    label_triggered = Signal(int, str, float)

    def __init__(
        self,
        labels: Optional[list[tuple[str, int, str, str]]] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._labels = labels or DEFAULT_LABELS

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 4, 2, 4)

        grid = QGridLayout()
        grid.setSpacing(4)

        for i, (name, code, key, color) in enumerate(self._labels):
            btn = QPushButton(f"{name}\n[{key}]")
            btn.setMinimumHeight(38)
            btn.setStyleSheet(
                f"QPushButton {{ "
                f"background-color: {color}; color: white; "
                f"font-weight: bold; border-radius: 4px; padding: 4px; "
                f"font-size: 12px; "
                f"}} "
                f"QPushButton:hover {{ opacity: 0.8; }} "
                f"QPushButton:pressed {{ border: 3px solid white; }}"
            )
            btn.clicked.connect(
                lambda checked, c=code, n=name: self._emit_label(c, n)
            )
            grid.addWidget(btn, i // 4, i % 4)

            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(
                lambda c=code, n=name: self._emit_label(c, n)
            )

        layout.addLayout(grid)

        self._last_label = QLabel("当前标记: —")
        self._last_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._last_label)

    def _emit_label(self, code: int, name: str):
        ts = time.time()
        self.label_triggered.emit(code, name, ts)
        self._last_label.setText(f"当前标记: [{code}] {name}")
        self._last_label.setStyleSheet(
            "color: #FFD700; font-weight: bold; font-size: 12px;"
        )


# 兼容旧名
DualEEGEMGWidget = MultiRegionEEGWidget
