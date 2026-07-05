# -*- coding: utf-8 -*-
"""
吞咽评估第二范式。

每个 epoch:
  静息1(5s) -> 想象吞咽1(5s)

该脚本与 paradigm_swallow.py 同级，复用相同的显示、音频和 Marker
后端，并通过 stdout 输出 EVENT:... 供 GUI 写入 EEG/EMG/ECG 录制标签。
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


EVENTS = [
    {
        "name": "静息1",
        "delay": 5.0,
        "sound": "静息",
        "subtitle": "请保持放松，静息状态",
    },
    {
        "name": "想象吞咽1",
        "delay": 5.0,
        "sound": "想象吞咽",
        "subtitle": "请想象吞咽动作",
    },
]


def _show_start_screen(display) -> bool:
    display.show_message(
        title="吞咽评估第二范式",
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
        if display.poll_key() == "escape":
            raise ExperimentAborted("用户按 ESC 退出")
        time.sleep(0.05)


def _emit_event(marker_sender, label: str) -> None:
    if marker_sender is not None:
        marker_sender.send(label)
    print(f"EVENT:{label}", flush=True)


def run_part2_paradigm(
    patient_id: str,
    epoch_count: int = 10,
    experiment_date: str = "",
    marker_sender: Optional[LSLMarkerSender] = None,
    display=None,
    audio_player=None,
    log_dir: str = "logs",
    debug: bool = False,
) -> list:
    exp_date = experiment_date or datetime.date.today().isoformat()
    log_path = Path(log_dir) / patient_id / "cupture_data_part2"
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filepath = log_path / f"swallow_part2_log_{timestamp_str}.csv"
    print(f"[Log] 第二范式日志文件: {csv_filepath}", flush=True)
    print(f"[Paradigm] 实验日期: {exp_date}", flush=True)
    print(f"[Paradigm] 预计总时长: ~{epoch_count * 10:.0f}s ({epoch_count} epoch(s))", flush=True)

    csv_file = open(str(csv_filepath), "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "event_index", "event_name",
        "timestamp_sec", "system_time",
    ])

    log_records = []
    event_counter = 1

    try:
        ts = time.time()
        sys_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        _emit_event(marker_sender, "实验开始")
        log_records.append({
            "epoch": 0,
            "event_index": event_counter,
            "event_name": "实验开始",
            "timestamp_sec": ts,
            "system_time": sys_time,
        })
        csv_writer.writerow([
            0, event_counter, "实验开始", f"{ts:.6f}", sys_time,
        ])

        for epoch in range(1, epoch_count + 1):
            print(f"\n{'=' * 50}", flush=True)
            print(f"Part2 Epoch {epoch}/{epoch_count}", flush=True)
            print(f"{'=' * 50}", flush=True)

            for evt in EVENTS:
                event_counter += 1
                epoch_label = f"E{epoch}_{evt['name']}" if epoch_count > 1 else evt["name"]
                ts = time.time()
                sys_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                _emit_event(marker_sender, epoch_label)

                if debug:
                    print(
                        f"  [Event {event_counter}] {epoch_label} "
                        f"delay={evt['delay']}s sound='{evt['sound']}' ts={ts:.3f}",
                        flush=True,
                    )

                title = "吞咽评估第二范式"
                if epoch_count > 1:
                    title += f" (Epoch {epoch}/{epoch_count})"
                display.show_message(
                    title=title,
                    subtitle=evt["subtitle"],
                    footer="请保持放松 | 按【ESC】键退出实验",
                )

                audio_file = SOUND_FILE_MAP.get(evt["sound"], "")
                if audio_file and audio_player and audio_player.is_available():
                    try:
                        audio_player.play(audio_file, wait=True)
                    except Exception as exc:
                        print(f"[Audio] 播放异常: {exc}", flush=True)

                _wait_with_escape(display, float(evt["delay"]))

                record = {
                    "epoch": epoch,
                    "event_index": event_counter,
                    "event_name": epoch_label,
                    "timestamp_sec": ts,
                    "system_time": sys_time,
                }
                log_records.append(record)
                csv_writer.writerow([
                    epoch, event_counter, epoch_label, f"{ts:.6f}", sys_time,
                ])
                csv_file.flush()

        event_counter += 1
        ts = time.time()
        sys_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        _emit_event(marker_sender, "实验结束")
        log_records.append({
            "epoch": 0,
            "event_index": event_counter,
            "event_name": "实验结束",
            "timestamp_sec": ts,
            "system_time": sys_time,
        })
        csv_writer.writerow([
            0, event_counter, "实验结束", f"{ts:.6f}", sys_time,
        ])
        csv_file.flush()
        display.show_message(
            title="吞咽评估第二范式",
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
        print("\n[Paradigm] 第二范式已被用户中止（ESC键）", flush=True)
        ts = time.time()
        sys_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_records.append({
            "epoch": -1,
            "event_index": event_counter + 1,
            "event_name": "实验中止",
            "timestamp_sec": ts,
            "system_time": sys_time,
        })
        csv_writer.writerow([
            -1, event_counter + 1, "实验中止", f"{ts:.6f}", sys_time,
        ])
        if marker_sender is not None:
            marker_sender.send("实验中止")
    finally:
        csv_file.close()
        print(f"[Log] 第二范式日志已保存: {csv_filepath}", flush=True)

    return log_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="吞咽评估第二范式 — 静息5秒 + 想象吞咽5秒",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--patient-id", type=str, required=True, help="患者编号")
    parser.add_argument("--epoch-count", type=int, default=10, help="重复次数，默认 10")
    parser.add_argument("--experiment-date", type=str, default="", help="实验日期")
    parser.add_argument("--eeg-marker", action="store_true", help="启用 LSL Marker 输出")
    parser.add_argument(
        "--marker-stream-name", type=str,
        default="swallow_part2_markers",
        help="LSL Marker 流名称",
    )
    parser.add_argument("--marker-udp-host", type=str, default=None, help="UDP Marker 主机")
    parser.add_argument("--marker-udp-port", type=int, default=None, help="UDP Marker 端口")
    parser.add_argument("--debug", action="store_true", help="调试模式")
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
        run_part2_paradigm(
            patient_id=args.patient_id,
            epoch_count=args.epoch_count,
            experiment_date=args.experiment_date,
            marker_sender=marker_sender,
            display=display,
            audio_player=audio_player,
            log_dir=log_dir,
            debug=args.debug,
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
