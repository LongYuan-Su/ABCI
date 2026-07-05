"""多模态模拟信号生成器 — 基于 ``metabci.brainda.datasets.base.BaseDataset``。

生成包含脑电(EEG)、心电(ECG)、肌电(EMG)的合成数据流，
模拟吞咽事件的时序标记，用于闭环系统离线测试。

继承 ``brainda.datasets.base.BaseDataset`` 以兼容 brainda 数据管线。

输出格式: (n_channels, n_samples) NumPy 数组 + 事件标记列表
"""

import numpy as np
from scipy import signal

from ..base import BaseDataset


class SwallowDataSimulator(BaseDataset):
    """吞咽多模态数据模拟器 — 继承 ``brainda.datasets.base.BaseDataset``。

    生成 16 通道合成信号:
    - EEG 通道 (9): P300 响应 + alpha 背景
    - EMG 通道 (6): 吞咽肌电爆发
    - ECG 通道 (1): 周期性心跳

    通过 ``_get_single_subject_data()`` 返回标准 mne RawArray，
    可被 ``brainda.paradigms`` 管线直接消费。

    Parameters
    ----------
    srate : float
        采样率 (Hz)，默认 250。
    duration : float
        数据总时长 (秒)，默认 60。
    noise_level : float
        背景噪声标准差，默认 1.0。
    """

    def __init__(
        self,
        srate: float = 250.0,
        duration: float = 60.0,
        noise_level: float = 1.0,
        n_channels: int = 16,
    ):
        # BaseDataset initialisation for brainda pipeline compatibility
        super().__init__(
            dataset_code="simulated_swallow",
            subjects=[1],
            events={
                "p300": (1, (0.0, 0.8)),
                "swallow": (2, (0.0, 1.5)),
            },
            channels=[f"Ch{i + 1}" for i in range(n_channels)],
            srate=srate,
            paradigm="swallow_imagery",
        )

        self._srate_val = srate
        self.duration = duration
        self.noise_level = noise_level
        self.n_channels = n_channels
        self.n_samples = int(srate * duration)

        # 存储事件
        self.events = []  # [(sample_index, event_label), ...]

    def _get_single_subject_data(self, subject_idx: int):
        """BaseDataset override: return simulated data as mne RawArray.

        Generates synthetic multi-modal data and wraps it in an
        ``mne.io.RawArray`` with proper channel info.
        """
        try:
            import mne
            data = self.generate()
            info = mne.create_info(
                ch_names=self.channels,
                sfreq=self._srate_val,
                ch_types=["eeg"] * self.n_channels,
            )
            return mne.io.RawArray(data, info)
        except ImportError:
            raise NotImplementedError(
                "mne is required for BaseDataset compatibility. "
                "Use generate() / to_epochs() for numpy-only access."
            )

    def _generate_eeg_1ch(self, p300_times: list = None) -> np.ndarray:
        """生成单通道模拟 EEG 信号（含 P300 响应），作为主通道模板。

        P300 特征：在事件发生后约 300ms 出现正波峰。

        Parameters
        ----------
        p300_times : list of float or None
            P300 事件发生的时间点（秒），None 则随机生成。

        Returns
        -------
        eeg : ndarray, shape (n_samples,)
        p300_times : list of float
            实际使用的 P300 事件时间点。
        """
        eeg = np.random.randn(self.n_samples) * self.noise_level

        # 添加 alpha 频段背景 (8-13 Hz)
        t = np.arange(self.n_samples) / self.srate
        eeg += 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha

        # 如果没有指定 P300 时间，则随机生成 10 个
        if p300_times is None:
            n_events = 10
            p300_times = np.sort(
                np.random.uniform(5, self.duration - 5, n_events)
            )

        # 在每个 P300 事件处添加 P300 波形
        for et in p300_times:
            sample_idx = int(et * self.srate)
            # P300 波形: 250-450ms 处出现正波
            p300_peak_offset = int(0.30 * self.srate)  # 300ms 延迟
            p300_width = int(0.20 * self.srate)  # 200ms 宽度
            peak_idx = sample_idx + p300_peak_offset
            if peak_idx < self.n_samples:
                end_idx = min(peak_idx + p300_width * 2, self.n_samples)
                length = end_idx - peak_idx
                # 高斯形状的 P300 波
                gauss = 3.0 * np.exp(
                    -0.5 * ((np.arange(length) - p300_width) / (p300_width / 3)) ** 2
                )
                eeg[peak_idx:end_idx] += gauss

            # 记录事件
            self.events.append((sample_idx, 1))  # label=1 代表 P300 事件

        return eeg, p300_times

    def generate_eeg(self, p300_times: list = None) -> np.ndarray:
        """生成多通道模拟 EEG 信号（含 P300 响应）。

        通道 0 为主通道（含完整 P300 波形），其余通道为空间衰减变体，
        模拟真实导联间信号的空间相关性。

        Parameters
        ----------
        p300_times : list of float or None
            P300 事件发生的时间点（秒），None 则随机生成。

        Returns
        -------
        eeg : ndarray, shape (n_channels, n_samples)
        """
        # 生成主通道（模板）
        eeg_template, p300_times = self._generate_eeg_1ch(p300_times)

        if self.n_channels == 1:
            return eeg_template[np.newaxis, :]

        # 生成多通道：每个通道是主通道的衰减副本 + 独立噪声
        eeg_multi = np.zeros((self.n_channels, self.n_samples))
        eeg_multi[0] = eeg_template

        for ch in range(1, self.n_channels):
            # 衰减系数 0.5~0.95，模拟不同导联的 P300 幅度差异
            attenuation = np.random.uniform(0.5, 0.95)
            # 独立背景噪声
            ch_noise = np.random.randn(self.n_samples) * self.noise_level * 0.5
            eeg_multi[ch] = eeg_template * attenuation + ch_noise

        return eeg_multi

    def generate_ecg(self) -> np.ndarray:
        """生成模拟 ECG 信号（周期性心跳）。

        心率约 72 bpm，每个心跳包含 P-QRS-T 波形。

        Returns
        -------
        ecg : ndarray, shape (n_samples,)
        """
        ecg = np.zeros(self.n_samples)
        heart_rate = 72  # bpm
        beat_interval = int(self.srate * 60 / heart_rate)  # 采样点间隔

        # 简化版 QRS 波形
        qrs_dur = int(0.08 * self.srate)  # 80ms QRS 宽度
        qrs = np.zeros(beat_interval)
        mid = beat_interval // 2
        half_w = qrs_dur // 2
        if mid - half_w >= 0 and mid + half_w < beat_interval:
            qrs[mid - half_w : mid] = np.linspace(0, 1.5, half_w)  # R 波上升
            qrs[mid : mid + half_w] = np.linspace(1.5, -0.5, half_w)  # R 波下降

        # 重复心跳
        n_beats = self.n_samples // beat_interval + 1
        for i in range(n_beats):
            start = i * beat_interval
            end = min(start + beat_interval, self.n_samples)
            ecg[start:end] = qrs[: end - start]

        ecg += np.random.randn(self.n_samples) * 0.05  # 少量噪声
        return ecg

    def generate_emg(self, swallow_times: list = None) -> np.ndarray:
        """生成模拟 EMG 信号（吞咽时产生肌电爆发）。

        Parameters
        ----------
        swallow_times : list of float or None
            吞咽事件发生的时间点（秒）。

        Returns
        -------
        emg : ndarray, shape (n_samples,)
        """
        emg = np.random.randn(self.n_samples) * 0.1  # 基线噪声极低

        if swallow_times is None:
            n_swallows = 8
            swallow_times = np.sort(
                np.random.uniform(5, self.duration - 5, n_swallows)
            )

        for st in swallow_times:
            sample_idx = int(st * self.srate)
            # 肌电爆发: 持续约 1 秒，高频高幅
            burst_dur = int(1.0 * self.srate)
            end_idx = min(sample_idx + burst_dur, self.n_samples)
            length = end_idx - sample_idx
            if length > 0:
                # 高频噪声 (50-150 Hz 模拟)
                t_burst = np.arange(length) / self.srate
                burst = np.random.randn(length) * 3.0
                # 包络：快速上升，缓慢下降
                envelope = np.exp(-np.linspace(0, 3, length))
                emg[sample_idx:end_idx] += burst * envelope

            # 记录事件
            self.events.append((sample_idx, 2))  # label=2 代表吞咽事件

        return emg

    def generate(self, p300_times: list = None, swallow_times: list = None) -> dict:
        """生成完整的多模态数据集。

        Parameters
        ----------
        p300_times : list of float or None
            P300 事件时间点。
        swallow_times : list of float or None
            吞咽事件时间点。

        Returns
        -------
        data : dict
            {
                "data": ndarray shape (n_channels, n_samples) — 多通道EEG,
                "srate": float — 采样率,
                "events": list of (sample_idx, label) — 事件标记,
                "channels": list of str — 通道名称列表
            }
        """
        self.events = []

        # 生成多通道 EEG 数据（含 P300 事件）
        eeg_data = self.generate_eeg(p300_times)

        # 生成 EMG 信号并记录吞咽事件
        if swallow_times is not None:
            self.generate_emg(swallow_times)

        # 按时间排序事件
        self.events.sort(key=lambda x: x[0])

        # 生成通道名称
        channel_names = [f"EEG_{i+1}" for i in range(self.n_channels)]

        return {
            "data": eeg_data,
            "srate": self.srate,
            "events": self.events,
            "channels": channel_names,
        }

    def to_epochs(self, data: np.ndarray, tmin: float, tmax: float,
                  event_times: list) -> np.ndarray:
        """将连续数据切分为试次。

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            连续数据。
        tmin : float
            事件前时间（秒，负数）。
        tmax : float
            事件后时间（秒，正数）。
        event_times : list of float
            事件时间点列表（秒）。

        Returns
        -------
        epochs : ndarray, shape (n_events, n_channels, n_times)
        """
        n_channels = data.shape[0]
        n_times = int((tmax - tmin) * self.srate)
        epochs = np.zeros((len(event_times), n_channels, n_times))

        for i, et in enumerate(event_times):
            center = int(et * self.srate)
            start = center + int(tmin * self.srate)
            end = start + n_times
            if start >= 0 and end <= data.shape[1]:
                epochs[i] = data[:, start:end]

        return epochs


def main():
    """生成示例数据集并保存。"""
    import os

    np.random.seed(42)

    sim = SwallowDataSimulator(srate=250, duration=30, noise_level=0.5, n_channels=16)

    # 指定 P300 和吞咽事件时间
    p300_times = [5, 10, 15, 20, 25]
    swallow_times = [5.5, 10.5, 15.5, 20.5, 25.5]  # P300 后 0.5s 吞咽

    dataset = sim.generate(p300_times=p300_times, swallow_times=swallow_times)

    print(f"生成数据: {dataset['data'].shape}, 采样率: {dataset['srate']} Hz")
    print(f"通道数: {sim.n_channels}, 事件数: {len(dataset['events'])}")
    print(f"通道: {dataset['channels'][:5]}...")
    print(f"前 10 个事件: {dataset['events'][:10]}")

    # 保存数据
    output_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(output_dir, "simulated_data.npz")
    np.savez(
        save_path,
        data=dataset["data"],
        srate=dataset["srate"],
        events=np.array(dataset["events"]),
        channels=dataset["channels"],
    )
    print(f"\n数据已保存到: {save_path}")

    # 简单统计
    print(f"\n数据统计（前5通道）:")
    for ch in range(min(5, sim.n_channels)):
        print(f"  {dataset['channels'][ch]} 范围: "
              f"[{dataset['data'][ch].min():.2f}, {dataset['data'][ch].max():.2f}]")


if __name__ == "__main__":
    main()
