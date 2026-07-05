# -*- coding: utf-8 -*-
"""ESP32B electrical stimulation controller.

This module serves two purposes:
1. A reusable UDP client used by the swallow-control paradigm.
2. A small Tkinter GUI for manually checking the hardware link and toggling
   AQW214S channels.

Expected firmware commands:
    PING
    STATUS
    CH1 ON / CH1 OFF ... CH4 ON / CH4 OFF
    ALL ON / ALL OFF
"""

from __future__ import annotations

import argparse
import json
import queue
import socket
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any


DEFAULT_HOST = "192.168.4.1"
DEFAULT_PORT = 3333
REQUEST_TIMEOUT_SECONDS = 0.8
UDP_RETRY_COUNT = 3
HEARTBEAT_RETRY_COUNT = 1
HEARTBEAT_INTERVAL_MS = 2000
HEARTBEAT_FAIL_THRESHOLD = 3
HEARTBEAT_ERROR_LOG_INTERVAL_SECONDS = 15.0
UDP_REQUEST_BURST_COUNT = 2
UDP_REQUEST_BURST_GAP_SECONDS = 0.02
UDP_RESPONSE_SIZE = 1024
CHANNELS = ("CH1", "CH2", "CH3", "CH4")


def _parse_response(response: str) -> Any:
    response = response.strip()
    if response.startswith("{"):
        return json.loads(response)
    return response


@dataclass
class Esp32UdpController:
    """Small UDP client shared by the GUI and the closed-loop paradigm."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    timeout: float = REQUEST_TIMEOUT_SECONDS
    retries: int = UDP_RETRY_COUNT

    def _validate(self) -> None:
        if not self.host.strip():
            raise ValueError("ESP32B 地址不能为空")
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("端口范围必须是 1-65535")

    def send_once(self, command: str) -> Any:
        self._validate()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control_socket:
            control_socket.settimeout(self.timeout)
            payload = (command.strip() + "\n").encode("utf-8")
            for index in range(UDP_REQUEST_BURST_COUNT):
                control_socket.sendto(payload, (self.host.strip(), int(self.port)))
                if index + 1 < UDP_REQUEST_BURST_COUNT:
                    time.sleep(UDP_REQUEST_BURST_GAP_SECONDS)
            response_data, _ = control_socket.recvfrom(UDP_RESPONSE_SIZE)
        return _parse_response(response_data.decode("utf-8", errors="replace"))

    def send(self, command: str, retries: int | None = None) -> Any:
        retry_count = self.retries if retries is None else retries
        last_error: Exception | None = None
        for _ in range(max(1, retry_count)):
            try:
                return self.send_once(command)
            except (OSError, TimeoutError) as error:
                last_error = error
        raise TimeoutError(f"UDP 超时，已重试 {retry_count} 次：{last_error}")

    def ping(self) -> Any:
        return self.send("PING")

    def status(self) -> Any:
        return self.send("STATUS")

    def set_channel(self, target: str, enabled: bool) -> Any:
        target = target.upper().strip()
        if target != "ALL" and target not in CHANNELS:
            raise ValueError(f"未知通道：{target}")
        return self.send(f"{target} {'ON' if enabled else 'OFF'}")

    def trigger(self, target: str = "ALL", duration: float = 1.0) -> tuple[Any, Any | None]:
        """Turn stimulation on, optionally wait, then turn it off.

        Set duration <= 0 to keep the selected channel on until a manual OFF.
        """
        on_response = self.set_channel(target, True)
        off_response = None
        if duration > 0:
            time.sleep(duration)
            off_response = self.set_channel(target, False)
        return on_response, off_response


class Esp32ControlApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ESP32B 电刺激控制器")
        self.root.geometry("720x540")
        self.root.minsize(640, 480)

        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        self.port_var = tk.StringVar(value=str(DEFAULT_PORT))
        self.connection_var = tk.StringVar(value="UDP 未测试")
        self.trigger_channel_var = tk.StringVar(value="ALL")
        self.trigger_duration_var = tk.DoubleVar(value=1.0)
        self.channel_vars = {channel: tk.StringVar(value="未知") for channel in CHANNELS}
        self.worker_messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.socket_lock = threading.Lock()
        self.command_buttons: list[ttk.Button] = []
        self.command_busy = False
        self.monitoring_enabled = False
        self.heartbeat_busy = False
        self.last_heartbeat_ok: bool | None = None
        self.heartbeat_fail_count = 0
        self.last_heartbeat_error_log_time = 0.0

        self._build_ui()
        self._poll_worker_messages()
        self._schedule_heartbeat()

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)

        connection_frame = ttk.LabelFrame(root_frame, text="连接")
        connection_frame.pack(fill=tk.X)

        ttk.Label(connection_frame, text="ESP32B 地址").grid(row=0, column=0, padx=8, pady=8, sticky=tk.W)
        ttk.Entry(connection_frame, textvariable=self.host_var, width=18).grid(row=0, column=1, padx=8, pady=8, sticky=tk.W)
        ttk.Label(connection_frame, text="端口").grid(row=0, column=2, padx=8, pady=8, sticky=tk.W)
        ttk.Entry(connection_frame, textvariable=self.port_var, width=7).grid(row=0, column=3, padx=8, pady=8, sticky=tk.W)
        self._add_button(connection_frame, "连接测试", self.ping, 0, 4)
        ttk.Button(connection_frame, text="断开监测", command=self.disconnect).grid(row=0, column=5, padx=8, pady=8)
        self._add_button(connection_frame, "刷新状态", self.refresh_status, 0, 6)
        ttk.Label(connection_frame, textvariable=self.connection_var).grid(row=0, column=7, padx=8, pady=8, sticky=tk.W)
        connection_frame.columnconfigure(7, weight=1)

        channels_frame = ttk.LabelFrame(root_frame, text="AQW214S 四路手动控制")
        channels_frame.pack(fill=tk.X, pady=(12, 0))

        for row, channel in enumerate(CHANNELS):
            ttk.Label(channels_frame, text=channel, width=8).grid(row=row, column=0, padx=8, pady=6, sticky=tk.W)
            ttk.Label(channels_frame, textvariable=self.channel_vars[channel], width=8).grid(row=row, column=1, padx=8, pady=6)
            self._add_button(channels_frame, "打开", lambda selected=channel: self.send_command(selected, "ON"), row, 2)
            self._add_button(channels_frame, "关闭", lambda selected=channel: self.send_command(selected, "OFF"), row, 3)

        all_frame = ttk.Frame(root_frame)
        all_frame.pack(fill=tk.X, pady=(12, 0))
        self._add_packed_button(all_frame, "全部打开", lambda: self.send_command("ALL", "ON"))
        self._add_packed_button(all_frame, "全部关闭", lambda: self.send_command("ALL", "OFF"))

        trigger_frame = ttk.LabelFrame(root_frame, text="自动触发测试")
        trigger_frame.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(trigger_frame, text="通道").grid(row=0, column=0, padx=8, pady=8, sticky=tk.W)
        ttk.Combobox(
            trigger_frame,
            textvariable=self.trigger_channel_var,
            values=("ALL",) + CHANNELS,
            width=8,
            state="readonly",
        ).grid(row=0, column=1, padx=8, pady=8, sticky=tk.W)
        ttk.Label(trigger_frame, text="持续时间(s)").grid(row=0, column=2, padx=8, pady=8, sticky=tk.W)
        ttk.Spinbox(
            trigger_frame,
            from_=0.0,
            to=60.0,
            increment=0.5,
            textvariable=self.trigger_duration_var,
            width=8,
        ).grid(row=0, column=3, padx=8, pady=8, sticky=tk.W)
        self._add_button(trigger_frame, "触发一次", self.trigger_once, 0, 4)
        ttk.Label(trigger_frame, text="持续时间为 0 时保持打开，需手动关闭。").grid(
            row=0, column=5, padx=8, pady=8, sticky=tk.W
        )

        log_frame = ttk.LabelFrame(root_frame, text="通信日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _add_button(self, parent, text: str, command, row: int, column: int) -> None:
        button = ttk.Button(parent, text=text, command=command)
        button.grid(row=row, column=column, padx=8, pady=8)
        self.command_buttons.append(button)

    def _add_packed_button(self, parent, text: str, command) -> None:
        button = ttk.Button(parent, text=text, command=command)
        button.pack(side=tk.LEFT, padx=(0, 8))
        self.command_buttons.append(button)

    def _controller(self) -> Esp32UdpController:
        try:
            port = int(self.port_var.get().strip())
        except ValueError as error:
            raise ValueError("端口必须是数字") from error
        return Esp32UdpController(host=self.host_var.get().strip(), port=port)

    def set_command_busy(self, busy: bool) -> None:
        self.command_busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        for button in self.command_buttons:
            button.configure(state=state)

    def enable_monitoring(self) -> None:
        if self.monitoring_enabled:
            return
        self.monitoring_enabled = True
        self.last_heartbeat_ok = None
        self.heartbeat_fail_count = 0

    def disconnect(self) -> None:
        self.monitoring_enabled = False
        self.last_heartbeat_ok = None
        self.heartbeat_fail_count = 0
        self.connection_var.set("UDP 已停止监测")
        self.append_log("已停止 UDP 自动监测")

    def run_worker(self, task_name: str, worker) -> None:
        if self.command_busy:
            self.append_log(f"{task_name}: 上一条命令还没完成，已忽略")
            return
        self.set_command_busy(True)

        def wrapped() -> None:
            try:
                result = worker()
                self.worker_messages.put(("success", (task_name, result)))
            except Exception as error:
                self.worker_messages.put(("error", (task_name, str(error))))

        threading.Thread(target=wrapped, daemon=True).start()

    def ping(self) -> None:
        self.enable_monitoring()
        self.run_worker("连接测试", lambda: self._controller().ping())

    def refresh_status(self) -> None:
        self.enable_monitoring()
        self.run_worker("刷新状态", lambda: self._controller().status())

    def send_command(self, target: str, action: str) -> None:
        self.enable_monitoring()
        self.run_worker(
            f"{target} {action}",
            lambda: self._controller().set_channel(target, action.upper() == "ON"),
        )

    def trigger_once(self) -> None:
        self.enable_monitoring()
        target = self.trigger_channel_var.get().strip() or "ALL"
        duration = float(self.trigger_duration_var.get())
        self.run_worker(
            f"触发 {target} {duration:.1f}s",
            lambda: self._controller().trigger(target=target, duration=duration),
        )

    def update_channel_status(self, status: dict[str, object]) -> None:
        for channel in CHANNELS:
            if channel in status:
                self.channel_vars[channel].set("打开" if status[channel] else "关闭")

    def update_channel_status_from_line(self, line: str) -> None:
        parts = line.strip().split()
        if len(parts) < 3 or parts[0].upper() != "OK":
            return
        target = parts[1].upper()
        action = parts[2].upper()
        if action not in {"ON", "OFF"}:
            return
        label = "打开" if action == "ON" else "关闭"
        if target == "ALL":
            for channel in CHANNELS:
                self.channel_vars[channel].set(label)
        elif target in CHANNELS:
            self.channel_vars[target].set(label)

    def handle_success(self, task_name: str, result: object) -> None:
        self.last_heartbeat_ok = True
        self.heartbeat_fail_count = 0
        self.connection_var.set("UDP 通信正常")
        self.append_log(f"{task_name}: {result}")
        if isinstance(result, dict):
            self.update_channel_status(result)
        elif isinstance(result, tuple):
            for item in result:
                if isinstance(item, dict):
                    self.update_channel_status(item)
                elif isinstance(item, str):
                    self.update_channel_status_from_line(item)
        elif isinstance(result, str):
            self.update_channel_status_from_line(result)

    def handle_error(self, task_name: str, error: str) -> None:
        self.last_heartbeat_ok = False
        self.heartbeat_fail_count = HEARTBEAT_FAIL_THRESHOLD
        self.connection_var.set("UDP 通信失败")
        self.append_log(f"{task_name}: 失败，{error}")

    def handle_heartbeat_success(self) -> None:
        if not self.monitoring_enabled:
            return
        if self.last_heartbeat_ok is False:
            self.append_log("自动监测: UDP 已恢复")
        self.last_heartbeat_ok = True
        self.heartbeat_fail_count = 0
        self.connection_var.set("UDP 通信正常")

    def handle_heartbeat_error(self, error: str) -> None:
        if not self.monitoring_enabled:
            return
        self.heartbeat_fail_count += 1
        if self.heartbeat_fail_count < HEARTBEAT_FAIL_THRESHOLD:
            return
        self.connection_var.set("UDP 暂无响应，继续重试中")
        now = time.monotonic()
        if (
            self.last_heartbeat_ok is not False
            or now - self.last_heartbeat_error_log_time >= HEARTBEAT_ERROR_LOG_INTERVAL_SECONDS
        ):
            self.append_log(f"自动监测: UDP 暂时无响应，继续重试，{error}")
            self.last_heartbeat_error_log_time = now
        self.last_heartbeat_ok = False

    def _poll_worker_messages(self) -> None:
        while True:
            try:
                message_type, payload = self.worker_messages.get_nowait()
            except queue.Empty:
                break
            task_name, data = payload
            if message_type == "success":
                self.handle_success(str(task_name), data)
                self.set_command_busy(False)
            elif message_type == "error":
                self.handle_error(str(task_name), str(data))
                self.set_command_busy(False)
            elif message_type == "heartbeat_success":
                self.heartbeat_busy = False
                self.handle_heartbeat_success()
            elif message_type == "heartbeat_error":
                self.heartbeat_busy = False
                self.handle_heartbeat_error(str(data))
        self.root.after(50, self._poll_worker_messages)

    def _schedule_heartbeat(self) -> None:
        self.root.after(HEARTBEAT_INTERVAL_MS, self._run_heartbeat)

    def _run_heartbeat(self) -> None:
        if self.monitoring_enabled and not self.command_busy and not self.heartbeat_busy:
            self.heartbeat_busy = True

            def wrapped() -> None:
                try:
                    self._controller().send("PING", retries=HEARTBEAT_RETRY_COUNT)
                    self.worker_messages.put(("heartbeat_success", ("自动心跳", None)))
                except Exception as error:
                    self.worker_messages.put(("heartbeat_error", ("自动心跳", str(error))))

            threading.Thread(target=wrapped, daemon=True).start()
        self._schedule_heartbeat()

    def append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ESP32B 电刺激控制器")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--command", choices=("gui", "ping", "status", "on", "off", "trigger"), default="gui")
    parser.add_argument("--target", default="ALL", help="ALL, CH1, CH2, CH3, CH4")
    parser.add_argument("--duration", type=float, default=1.0, help="trigger 持续时间；<=0 表示保持打开")
    return parser


def run_cli(args: argparse.Namespace) -> int:
    controller = Esp32UdpController(host=args.host, port=args.port)
    if args.command == "ping":
        result = controller.ping()
    elif args.command == "status":
        result = controller.status()
    elif args.command == "on":
        result = controller.set_channel(args.target, True)
    elif args.command == "off":
        result = controller.set_channel(args.target, False)
    elif args.command == "trigger":
        result = controller.trigger(args.target, args.duration)
    else:
        raise ValueError(f"未知命令：{args.command}")
    print(result, flush=True)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command != "gui":
        return run_cli(args)

    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except tk.TclError:
        pass
    app = Esp32ControlApp(root)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    try:
        app.refresh_status()
        root.mainloop()
    except KeyboardInterrupt:
        app.disconnect()
        root.destroy()
    except tk.TclError as error:
        messagebox.showerror("界面错误", str(error))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
