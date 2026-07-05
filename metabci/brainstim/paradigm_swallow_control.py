# -*- coding: utf-8 -*-
"""
吞咽意图调控范式。

流程：
  每个 epoch 持续提示想象吞咽 -> 算法识别吞咽意图
  -> 识别到意图后触发电刺激控制器预留接口 -> 进入下一轮

ESP32B 控制器接口已经接入；主 GUI 启动本范式时默认由实时 EEG
分类闭环触发电刺激。若命令行直接传入 --enable-controller，本脚本也可
按自身检测结果直接发送 ESP32B UDP 触发命令。
吞咽意图检测器优先加载 swallow_trainer 中的吞咽分类模型；如果未提供模型，
使用 cue 模式，让界面和事件流程可以先跑通。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import time
from pathlib import Path
from typing import Optional

_CURR_DIR = os.path.abspath(os.path.dirname(__file__))
_METABCI_DIR = os.path.abspath(os.path.join(_CURR_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_METABCI_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _METABCI_DIR not in sys.path:
    sys.path.insert(0, _METABCI_DIR)

from metabci.brainstim.paradigm_swallow import (  # noqa: E402
    ExperimentAborted,
    LSLMarkerSender,
    SOUND_FILE_MAP,
    create_audio_player,
    create_display_backend,
)
from applications.swallow_bci.tools.esp32b_control_gui import (  # noqa: E402
    DEFAULT_HOST as DEFAULT_CONTROLLER_HOST,
    DEFAULT_PORT as DEFAULT_CONTROLLER_PORT,
    Esp32UdpController,
)


CONTROL_EVENTS = [
    {
        "name": "想象吞咽",
        "delay": 5.0,
        "sound": "想象吞咽",
        "subtitle": "请持续想象吞咽动作",
    },
]


class SwallowIntentDetector:
    """吞咽意图检测适配器。

    后续接入实时 EEG/EMG/ECG 数据时，将 detect() 的 cue_mode 分支替换为
    模型输入预处理和 model(eeg, emg, ecg) 推理即可。
    """

    def __init__(
        self,
        model_path: str = "",
        threshold: float = 0.6,
        device: str = "cpu",
        cue_mode: bool = True,
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.device = device
        self.cue_mode = cue_mode
        self.model = None

        if model_path:
            try:
                from metabci.brainda.algorithms.deep_learning.swallow_trainer import (
                    load_swallow_classifier,
                )

                self.model = load_swallow_classifier(model_path, device=device)
                print(f"[Intent] 已加载吞咽分类模型: {model_path}", flush=True)
            except Exception as exc:
                print(f"[Intent] 模型加载失败，切换 cue 模式: {exc}", flush=True)
                self.model = None
                self.cue_mode = True

    def detect(self, event_name: str) -> dict:
        if self.model is None:
            detected = self.cue_mode and ("想象吞咽" in event_name)
            confidence = 1.0 if detected else 0.0
            return {
                "detected": detected,
                "confidence": confidence,
                "mode": "cue",
                "label": "想象吞咽" if detected else "非吞咽意图",
            }

        # Real-time data is not wired into this paradigm process yet.
        return {
            "detected": False,
            "confidence": 0.0,
            "mode": "model_ready_waiting_for_stream",
            "label": "等待实时数据",
        }


class SwallowAssistController:
    """电刺激辅助吞咽控制器接口。"""

    def __init__(
        self,
        enabled: bool = False,
        host: str = DEFAULT_CONTROLLER_HOST,
        port: int = DEFAULT_CONTROLLER_PORT,
        target: str = "ALL",
        duration: float = 1.0,
    ):
        self.enabled = enabled
        self.target = target
        self.duration = duration
        self.controller = Esp32UdpController(host=host, port=port)

    def trigger(self, confidence: float, label: str) -> str:
        if not self.enabled:
            print(
                f"CONTROL_PLACEHOLDER:检测到{label}, confidence={confidence:.2f}, "
                "控制器尚未接入，已预留触发点",
                flush=True,
            )
            return "placeholder_triggered"

        on_response, off_response = self.controller.trigger(
            target=self.target,
            duration=self.duration,
        )
        print(
            f"CONTROL_TRIGGER:{label}, confidence={confidence:.2f}, "
            f"target={self.target}, duration={self.duration:.2f}, "
            f"on={on_response}, off={off_response}",
            flush=True,
        )
        return "esp32_triggered"


def _show_start_screen(display) -> bool:
    display.show_message(
        title="吞咽意图调控范式",
        subtitle="实验即将开始，请保持放松\n\n按【空格键】开始实验",
        footer="按【ESC】键退出 | 按【空格键】开始",
    )
    display.clear_keys()
    while True:
        key = display.poll_key()
        if key == "space":
            return True
        if key == "escape":
            return False
        time.sleep(0.05)


def _wait_with_escape(display, seconds: float) -> None:
    start = time.time()
    while (time.time() - start) < seconds:
        key = display.poll_key()
        if key == "escape":
            raise ExperimentAborted("用户按 ESC 退出")
        time.sleep(0.05)


def run_control_paradigm(
    patient_id: str,
    epoch_count: int = 5,
    experiment_date: str = "",
    marker_sender: Optional[LSLMarkerSender] = None,
    display=None,
    audio_player=None,
    log_dir: str = "logs",
    debug: bool = False,
    model_path: str = "",
    intent_threshold: float = 0.6,
    controller_enabled: bool = False,
    controller_host: str = DEFAULT_CONTROLLER_HOST,
    controller_port: int = DEFAULT_CONTROLLER_PORT,
    controller_target: str = "ALL",
    controller_duration: float = 1.0,
    external_online_controller: bool = False,
) -> list:
    exp_date = experiment_date or datetime.date.today().isoformat()
    log_path = Path(log_dir) / patient_id
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filepath = log_path / f"swallow_control_log_{timestamp_str}.csv"
    print(f"[Log] 调控范式日志文件: {csv_filepath}", flush=True)
    print(f"[Paradigm] 实验日期: {exp_date}", flush=True)

    detector = SwallowIntentDetector(
        model_path=model_path,
        threshold=intent_threshold,
        cue_mode=not bool(model_path),
    )
    controller = SwallowAssistController(
        enabled=controller_enabled,
        host=controller_host,
        port=controller_port,
        target=controller_target,
        duration=controller_duration,
    )

    csv_file = open(str(csv_filepath), "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "event_index", "event_name", "timestamp_sec", "system_time",
        "intent_detected", "confidence", "detector_mode", "controller_action",
    ])

    log_records = []
    event_counter = 0

    try:
        for epoch in range(1, epoch_count + 1):
            print(f"\n{'=' * 50}", flush=True)
            print(f"Control Epoch {epoch}/{epoch_count}", flush=True)
            print(f"{'=' * 50}", flush=True)

            for evt in CONTROL_EVENTS:
                event_counter += 1
                name = evt["name"]
                epoch_label = f"E{epoch}_{name}" if epoch_count > 1 else name
                ts = time.time()
                sys_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                if marker_sender is not None:
                    marker_sender.send(epoch_label)
                print(f"EVENT:{epoch_label}", flush=True)

                title = "吞咽意图调控范式"
                if epoch_count > 1:
                    title += f" (Epoch {epoch}/{epoch_count})"
                display.show_message(
                    title=title,
                    subtitle=evt["subtitle"],
                    footer="请持续想象吞咽 | 按【ESC】键退出实验",
                )

                audio_file = SOUND_FILE_MAP.get(evt["sound"], "")
                if audio_file and audio_player and audio_player.is_available():
                    try:
                        audio_player.play(audio_file, wait=True)
                    except Exception as exc:
                        print(f"[Audio] 播放异常: {exc}", flush=True)

                _wait_with_escape(display, float(evt["delay"]))

                if external_online_controller:
                    result = {
                        "detected": False,
                        "confidence": 0.0,
                        "mode": "external_online_gui",
                        "label": "等待主GUI实时分类",
                    }
                    print(
                        "INTENT:等待主GUI实时分类,confidence=0.00,"
                        "detected=0,mode=external_online_gui",
                        flush=True,
                    )
                else:
                    result = detector.detect(epoch_label)
                controller_action = ""
                if "想象吞咽" in name:
                    intent_label = (
                        f"INTENT:想象吞咽,confidence={result['confidence']:.2f},"
                        f"detected={int(result['detected'])},mode={result['mode']}"
                    )
                    if not external_online_controller:
                        print(intent_label, flush=True)
                    if marker_sender is not None:
                        marker_sender.send("吞咽意图识别" if result["detected"] else "未识别吞咽意图")
                    if result["detected"] and result["confidence"] >= intent_threshold:
                        try:
                            controller_action = controller.trigger(
                                result["confidence"],
                                result["label"],
                            )
                        except Exception as exc:
                            controller_action = "esp32_trigger_failed"
                            print(f"CONTROL_ERROR:{exc}", flush=True)
                        display.show_message(
                            title=title,
                            subtitle=(
                                "算法识别到吞咽意图\n已触发电刺激控制器"
                                if controller_enabled
                                else "算法识别到吞咽意图\n已到达电刺激控制器预留接口"
                            ),
                            footer="按【ESC】键退出实验",
                        )
                        _wait_with_escape(display, 1.5 if debug else 2.0)

                record = {
                    "epoch": epoch,
                    "event_index": event_counter,
                    "event_name": epoch_label,
                    "timestamp_sec": ts,
                    "system_time": sys_time,
                    "intent_detected": int(result["detected"]),
                    "confidence": f"{result['confidence']:.6f}",
                    "detector_mode": result["mode"],
                    "controller_action": controller_action,
                }
                log_records.append(record)
                csv_writer.writerow([
                    epoch, event_counter, epoch_label, f"{ts:.6f}", sys_time,
                    int(result["detected"]), f"{result['confidence']:.6f}",
                    result["mode"], controller_action,
                ])
                csv_file.flush()

        display.show_message(
            title="吞咽意图调控范式",
            subtitle="实验已结束，感谢您的配合",
            footer="",
        )
        audio_file = SOUND_FILE_MAP.get("实验结束", "")
        if audio_file and audio_player and audio_player.is_available():
            try:
                audio_player.play(audio_file, wait=True)
            except Exception as exc:
                print(f"[Audio] 播放异常: {exc}", flush=True)
        _wait_with_escape(display, 1.5 if debug else 3.0)

    except ExperimentAborted:
        print("\n[Paradigm] 调控范式已被用户中止（ESC键）", flush=True)
        if marker_sender is not None:
            marker_sender.send("调控范式中止")
    finally:
        csv_file.close()
        print(f"[Log] 调控范式日志已保存: {csv_filepath}", flush=True)

    return log_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="吞咽意图调控范式 — 想象吞咽识别与控制器预留接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--patient-id", type=str, required=True, help="患者编号")
    parser.add_argument("--epoch-count", type=int, default=5, help="重复次数，默认 5")
    parser.add_argument("--experiment-date", type=str, default="", help="实验日期")
    parser.add_argument("--eeg-marker", action="store_true", help="启用 LSL Marker 输出")
    parser.add_argument(
        "--marker-stream-name", type=str,
        default="swallow_control_markers",
        help="LSL Marker 流名称",
    )
    parser.add_argument("--marker-udp-host", type=str, default=None, help="UDP Marker 主机")
    parser.add_argument("--marker-udp-port", type=int, default=None, help="UDP Marker 端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--classifier-model", type=str, default="", help="吞咽分类模型 .pt 路径")
    parser.add_argument("--intent-threshold", type=float, default=0.6, help="吞咽意图阈值")
    parser.add_argument(
        "--enable-controller", action="store_true",
        help="启用真实控制器触发分支；默认仅记录预留触发点",
    )
    parser.add_argument(
        "--external-online-controller", action="store_true",
        help="由主GUI实时分类闭环接管控制器触发；本脚本只负责提示和marker",
    )
    parser.add_argument("--controller-host", type=str, default=DEFAULT_CONTROLLER_HOST, help="ESP32B 控制器 IP")
    parser.add_argument("--controller-port", type=int, default=DEFAULT_CONTROLLER_PORT, help="ESP32B 控制器 UDP 端口")
    parser.add_argument("--controller-target", type=str, default="ALL", help="触发通道：ALL/CH1/CH2/CH3/CH4")
    parser.add_argument(
        "--controller-duration",
        type=float,
        default=1.0,
        help="电刺激打开持续时间；<=0 表示保持打开，需手动关闭",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.epoch_count < 1:
        print("错误: --epoch-count 必须 >= 1", flush=True)
        return 1

    assets_dir = os.path.join(_CURR_DIR, "assets")
    log_dir = os.path.join(_PROJECT_ROOT, "logs")

    display = create_display_backend(debug=args.debug)
    audio_player = create_audio_player(assets_dir)
    marker_sender = None
    if args.eeg_marker:
        marker_sender = LSLMarkerSender(
            stream_name=args.marker_stream_name,
            udp_host=args.marker_udp_host,
            udp_port=args.marker_udp_port,
        )

    try:
        if not _show_start_screen(display):
            return 0
        run_control_paradigm(
            patient_id=args.patient_id,
            epoch_count=args.epoch_count,
            experiment_date=args.experiment_date,
            marker_sender=marker_sender,
            display=display,
            audio_player=audio_player,
            log_dir=log_dir,
            debug=args.debug,
            model_path=args.classifier_model,
            intent_threshold=args.intent_threshold,
            controller_enabled=args.enable_controller,
            controller_host=args.controller_host,
            controller_port=args.controller_port,
            controller_target=args.controller_target,
            controller_duration=args.controller_duration,
            external_online_controller=args.external_online_controller,
        )
    finally:
        if marker_sender:
            marker_sender.close()
        if audio_player:
            audio_player.close()
        if display:
            display.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
