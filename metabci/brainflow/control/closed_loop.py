# -*- coding: utf-8 -*-
"""Swallow Closed-Loop Control System — P300 detection → confidence threshold → stimulation trigger.

Replicates metabci/apps/swallow_closed_loop.py using brainflow modules.

Usage:
    from metabci.brainflow.control.closed_loop import SwallowClosedLoop
    loop = SwallowClosedLoop()
    loop.train_with_simulation(60.0)
    loop.run(duration=30.0)
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

from metabci.brainflow.acquisition.sources import (
    RealTimeBuffer, SimSwallowAmplifier, OpenBCISource,
    WiFiShieldAmplifier, create_source,
)
from metabci.brainflow.control.stimulator import create_stimulator
from metabci.brainflow.logger import get_logger
from metabci.brainflow.processing.decoder import create_decoder

logger = get_logger("closed_loop")


class SwallowClosedLoop:
    """吞咽意图闭环调控系统。

    基于 brainflow 框架构建：

    Parameters
    ----------
    decoder_name : str
        解码器类型: "riemann", "eegnet", "lda".
    confidence_threshold : float
        吞咽意图置信度阈值 (0~1).
    stim_type : str
        刺激器类型: "print", "serial", "lsl".
    n_channels : int
        EEG 通道数.
    mode : str
        运行模式: "sim", "real", "wifi".
    srate : float
        采样率 (Hz).
    chunk_duration : float
        每次处理的数据块时长 (秒).
    p300_time_window : tuple
        P300 检测时间窗 (tmin, tmax).
    """

    def __init__(
        self,
        decoder_name: str = "riemann",
        decoder_kwargs: dict = None,
        confidence_threshold: float = 0.6,
        stim_type: str = "print",
        n_channels: int = 8,
        mode: str = "sim",
        srate: float = 250.0,
        chunk_duration: float = 0.5,
        p300_time_window: tuple = (0.0, 0.8),
    ):
        self.decoder_name = decoder_name
        self.decoder_kwargs = decoder_kwargs or {}
        self.confidence_threshold = confidence_threshold
        self.n_channels = n_channels
        self.mode = mode
        self.srate = srate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(srate * chunk_duration)
        self.p300_time_window = p300_time_window

        self.decoder = None
        self.stimulator = create_stimulator(stim_type)

        self.buffer = RealTimeBuffer(
            n_channels=n_channels, max_size=int(srate * 30)
        )

        self.total_triggers = 0
        self.total_chunks = 0
        self.detected_swallows = 0

        logger.info(
            "SwallowClosedLoop 初始化: mode=%s, decoder=%s, n_channels=%d, threshold=%.2f",
            mode, decoder_name, n_channels, confidence_threshold,
        )

    def train_with_simulation(self, duration: float = 60.0):
        """使用模拟数据训练解码器。"""
        from metabci.brainflow.acquisition.simulator import SwallowDataSimulator

        print(f"\n{'='*50}")
        print(f"训练阶段: 生成 {duration}s 模拟数据...")
        print(f"解码器: {self.decoder_name}, 通道数: {self.n_channels}")
        print(f"{'='*50}")

        sim = SwallowDataSimulator(
            srate=self.srate, duration=duration, n_channels=self.n_channels
        )
        n_events = int(duration / 5)
        p300_times = np.linspace(5, duration - 5, n_events)
        dataset = sim.generate(p300_times=p300_times)

        # Use only EEG channels (first n_channels) from the full multi-modal data
        eeg_data = dataset["data"][:self.n_channels, :]

        epochs = sim.to_epochs(
            eeg_data,
            tmin=self.p300_time_window[0],
            tmax=self.p300_time_window[1],
            event_times=p300_times,
        )

        X = epochs
        y = np.ones(len(epochs), dtype=int)

        n_nontarget = n_events * 4
        n_times = X.shape[2]
        X_nontarget = np.random.randn(n_nontarget, self.n_channels, n_times) * 0.3
        y_nontarget = np.zeros(n_nontarget, dtype=int)

        X_train = np.concatenate([X, X_nontarget], axis=0)
        y_train = np.concatenate([y, y_nontarget], axis=0)

        idx = np.random.permutation(len(X_train))
        X_train, y_train = X_train[idx], y_train[idx]

        print(f"训练数据: {X_train.shape}, 目标占比: {y_train.mean():.1%}")

        if self.decoder_name == "eegnet":
            self.decoder = create_decoder(
                "eegnet",
                n_channels=X_train.shape[1],
                n_samples=X_train.shape[2],
                n_classes=2,
                max_epochs=30,
                verbose=False,
            )
        else:
            self.decoder = create_decoder(self.decoder_name)

        self.decoder.fit(X_train, y_train)

        pred = self.decoder.predict(X_train)
        acc = (pred == y_train).mean()
        print(f"训练准确率: {acc:.2%}")
        print("训练完成。\n")
        logger.info("训练完成, 准确率: %.2f%%", acc * 100)

    def process_chunk(self, chunk: np.ndarray) -> dict:
        """处理一个数据块，检测吞咽意图。"""
        # Ensure chunk has correct number of channels
        if chunk.shape[0] != self.n_channels:
            return {"swallow_detected": False, "confidence": 0.0, "prediction": 0}

        # Check decoder readiness
        if self.decoder is None or not getattr(self.decoder, "is_fitted", False):
            return {"swallow_detected": False, "confidence": 0.0, "prediction": 0}

        eeg_input = chunk[np.newaxis, :, :]  # (1, n_channels, n_samples)

        proba = self.decoder.predict_proba(eeg_input)[0]
        pred = int(np.argmax(proba))
        confidence = proba[pred]

        swallow_detected = (pred == 1) and (confidence >= self.confidence_threshold)

        return {
            "swallow_detected": swallow_detected,
            "confidence": confidence,
            "prediction": pred,
        }

    def run(self, duration: float = 30.0, stream_name: str = "OpenBCI_GUI",
            host: str = "192.168.4.1", port: int = 9000):
        """运行闭环调控主循环。"""
        mode_labels = {
            "sim": "离线仿真",
            "real": "OpenBCI LSL 实时采集",
            "lsl": "OpenBCI LSL 实时采集",
            "wifi": "BrainFlow WiFi 直连",
        }
        mode_label = mode_labels.get(self.mode, self.mode)
        print(f"\n{'='*50}")
        print(f"闭环调控运行中... [{mode_label}]")
        print(f"时长: {duration}s, 通道: {self.n_channels}, "
              f"置信度阈值: {self.confidence_threshold:.0%}")
        print(f"{'='*50}\n")

        if self.mode == "sim":
            self._run_simulation(duration)
        elif self.mode == "wifi":
            self._run_wifi(duration, host, port)
        else:
            self._run_realtime(duration, stream_name)

    def _run_simulation(self, duration: float):
        """仿真模式。"""
        amplifier = SimSwallowAmplifier(
            srate=self.srate, n_channels=self.n_channels,
            duration=duration, chunk_size=self.chunk_samples,
        )
        dataset = amplifier.generate()
        full_data = dataset["data"]
        events = sorted(dataset["events"], key=lambda x: x[0])

        chunk_samples = self.chunk_samples
        n_chunks = full_data.shape[1] // chunk_samples
        window_samples = int(self.srate * 0.8)

        start_time = time.time()

        for i in range(n_chunks):
            start_idx = i * chunk_samples
            end_idx = start_idx + chunk_samples
            chunk = full_data[:, start_idx:end_idx]

            chunk_events = [
                (s - start_idx, l) for s, l in events
                if start_idx <= s < end_idx
            ]
            self.buffer.push(chunk, chunk_events)
            self.total_chunks += 1

            # Only process when buffer has enough samples
            if self.buffer.n_samples < window_samples:
                continue

            recent = self.buffer.get_recent(window_samples)
            if recent is None or recent.shape[1] < window_samples:
                continue

            result = self.process_chunk(recent)

            if result["swallow_detected"]:
                self.detected_swallows += 1
                self.stimulator.trigger(duration=0.5, intensity=result["confidence"])
                self.total_triggers += 1

            if i % 20 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[{elapsed:5.1f}s] 已处理: {i}/{n_chunks} 块, "
                    f"检测: {self.detected_swallows} 次, "
                    f"触发: {self.total_triggers} 次, "
                    f"置信度: {result['confidence']:.2f}"
                )

        elapsed = time.time() - start_time
        self._print_summary(elapsed)

    def _run_realtime(self, duration: float, stream_name: str):
        """LSL 实时模式。"""
        source = OpenBCISource(
            stream_name=stream_name, n_channels=self.n_channels, srate=self.srate,
        )
        source.start()

        print(f"等待 LSL 流 '{stream_name}' ...")
        t0 = time.time()
        while not source.is_connected():
            if time.time() - t0 > 15.0:
                print("错误: LSL 流连接超时")
                source.stop()
                return
            time.sleep(0.2)
        print(f"已连接: {source.get_n_channels()}ch, {source.get_srate():.0f}Hz\n")

        start_time = time.time()
        i = 0
        window_samples = int(self.srate * 0.8)
        while (time.time() - start_time) < duration:
            chunk = source.get_recent(self.chunk_samples)
            if chunk is None:
                time.sleep(0.01)
                continue
            markers = source.pop_markers()
            chunk_events = [(0, m[1]) for m in markers]
            self.buffer.push(chunk, chunk_events)
            self.total_chunks += 1
            i += 1

            if self.buffer.n_samples < window_samples:
                continue
            recent = self.buffer.get_recent(window_samples)
            if recent is None or recent.shape[1] < window_samples:
                continue
            result = self.process_chunk(recent)
            if result["swallow_detected"]:
                self.detected_swallows += 1
                self.stimulator.trigger(duration=0.5, intensity=result["confidence"])
                self.total_triggers += 1
            if i % 20 == 0:
                elapsed = time.time() - start_time
                print(f"[{elapsed:5.1f}s] 已处理: {i} 块, 检测: {self.detected_swallows} 次, "
                      f"触发: {self.total_triggers} 次, 置信度: {result['confidence']:.2f}")

        source.stop()
        elapsed = time.time() - start_time
        self._print_summary(elapsed)

    def _run_wifi(self, duration: float, host: str, port: int):
        """WiFi 直连模式。"""
        amplifier = WiFiShieldAmplifier(
            host=host, port=port, n_channels=self.n_channels, srate=self.srate,
        )
        amplifier.start()
        print(f"已连接 WiFi Shield: {host}:80 TCP:{port}, "
              f"{amplifier.get_n_channels()}ch, {amplifier.get_srate():.0f}Hz\n")

        start_time = time.time()
        i = 0
        window_samples = int(self.srate * 0.8)
        while (time.time() - start_time) < duration:
            chunk = amplifier.get_recent(self.chunk_samples)
            if chunk is None:
                time.sleep(0.01)
                continue
            self.buffer.push(chunk)
            self.total_chunks += 1
            i += 1

            if self.buffer.n_samples < window_samples:
                continue
            recent = self.buffer.get_recent(window_samples)
            if recent is None or recent.shape[1] < window_samples:
                continue
            result = self.process_chunk(recent)
            if result["swallow_detected"]:
                self.detected_swallows += 1
                self.stimulator.trigger(duration=0.5, intensity=result["confidence"])
                self.total_triggers += 1
            if i % 20 == 0:
                elapsed = time.time() - start_time
                print(f"[{elapsed:5.1f}s] 已处理: {i} 块, 检测: {self.detected_swallows} 次, "
                      f"触发: {self.total_triggers} 次, 置信度: {result['confidence']:.2f}")

        amplifier.stop()
        elapsed = time.time() - start_time
        self._print_summary(elapsed)

    def _print_summary(self, elapsed: float):
        print(f"\n{'='*50}")
        print(f"闭环运行完成！总时长: {elapsed:.1f}s")
        print(f"处理数据块: {self.total_chunks}")
        print(f"检测到吞咽意图: {self.detected_swallows} 次")
        print(f"触发刺激: {self.total_triggers} 次")
        print(f"{'='*50}")

    def save_model(self, filepath: str):
        """保存训练好的解码器模型。"""
        if self.decoder is None or not getattr(self.decoder, "is_fitted", False):
            raise RuntimeError("模型尚未训练，无法保存。")
        save_data = {
            "decoder_name": self.decoder_name,
            "n_channels": self.n_channels,
            "srate": self.srate,
        }
        with open(filepath, "wb") as f:
            pickle.dump({"model": self.decoder, "meta": save_data}, f)
        logger.info("模型已保存: %s", filepath)

    def load_model(self, filepath: str):
        """从文件加载预训练模型。"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.decoder = data["model"]
        if hasattr(self.decoder, "is_fitted"):
            self.decoder.is_fitted = True
        meta = data.get("meta", {})
        if "n_channels" in meta:
            self.n_channels = meta["n_channels"]
        if "srate" in meta:
            self.srate = meta["srate"]
        logger.info("模型已加载: %s", filepath)


def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="吞咽意图闭环调控系统 (brainflow)")
    parser.add_argument("--mode", type=str, default="sim", choices=["sim", "real", "wifi"])
    parser.add_argument("--decoder", type=str, default="riemann", choices=["riemann", "eegnet", "lda"])
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--n-channels", type=int, default=16)
    parser.add_argument("--stim", type=str, default="print", choices=["print", "serial", "lsl"])
    parser.add_argument("--stream-name", type=str, default="OpenBCI_GUI")
    parser.add_argument("--host", type=str, default="192.168.4.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--save-model", type=str, default=None)
    parser.add_argument("--load-model", type=str, default=None)
    args = parser.parse_args()

    system = SwallowClosedLoop(
        decoder_name=args.decoder,
        confidence_threshold=args.threshold,
        stim_type=args.stim,
        n_channels=args.n_channels,
        mode=args.mode,
    )

    if args.load_model:
        system.load_model(args.load_model)
        print(f"已加载预训练模型: {args.load_model}")
    elif args.mode == "sim":
        system.train_with_simulation(duration=60.0)
        if args.save_model:
            system.save_model(args.save_model)
    else:
        print("真实模式请使用 --load-model 加载预训练模型")
        return

    system.run(duration=args.duration, stream_name=args.stream_name,
               host=args.host, port=args.port)
    print("\n系统退出。")


if __name__ == "__main__":
    main()
