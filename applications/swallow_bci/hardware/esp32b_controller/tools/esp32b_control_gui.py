import json
import queue
import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk


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


class Esp32ControlApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ESP32B WiFi 控制上位机")
        self.root.geometry("640x470")
        self.root.minsize(580, 430)

        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        self.port_var = tk.StringVar(value=str(DEFAULT_PORT))
        self.connection_var = tk.StringVar(value="UDP 未测试")
        self.channel_vars = {
            channel: tk.StringVar(value="未知")
            for channel in CHANNELS
        }
        self.worker_messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.socket_lock = threading.Lock()
        self.control_socket: socket.socket | None = None
        self.socket_reader = None
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
        connect_button = ttk.Button(connection_frame, text="连接", command=self.ping)
        connect_button.grid(row=0, column=4, padx=8, pady=8)
        self.command_buttons.append(connect_button)
        ttk.Button(connection_frame, text="断开", command=self.disconnect).grid(row=0, column=5, padx=8, pady=8)
        refresh_button = ttk.Button(connection_frame, text="刷新状态", command=self.refresh_status)
        refresh_button.grid(row=0, column=6, padx=8, pady=8)
        self.command_buttons.append(refresh_button)
        ttk.Label(connection_frame, textvariable=self.connection_var).grid(row=0, column=7, padx=8, pady=8, sticky=tk.W)
        connection_frame.columnconfigure(7, weight=1)

        channels_frame = ttk.LabelFrame(root_frame, text="AQW214S 四路控制")
        channels_frame.pack(fill=tk.X, pady=(12, 0))

        for row, channel in enumerate(CHANNELS):
            ttk.Label(channels_frame, text=channel, width=8).grid(row=row, column=0, padx=8, pady=6, sticky=tk.W)
            ttk.Label(channels_frame, textvariable=self.channel_vars[channel], width=8).grid(row=row, column=1, padx=8, pady=6)
            on_button = ttk.Button(
                channels_frame,
                text="接通",
                command=lambda selected=channel: self.send_command(selected, "ON"),
            )
            on_button.grid(row=row, column=2, padx=8, pady=6)
            self.command_buttons.append(on_button)

            off_button = ttk.Button(
                channels_frame,
                text="断开",
                command=lambda selected=channel: self.send_command(selected, "OFF"),
            )
            off_button.grid(row=row, column=3, padx=8, pady=6)
            self.command_buttons.append(off_button)

        all_frame = ttk.Frame(root_frame)
        all_frame.pack(fill=tk.X, pady=(12, 0))
        all_on_button = ttk.Button(all_frame, text="全部接通", command=lambda: self.send_command("ALL", "ON"))
        all_on_button.pack(side=tk.LEFT, padx=(0, 8))
        self.command_buttons.append(all_on_button)
        all_off_button = ttk.Button(all_frame, text="全部断开", command=lambda: self.send_command("ALL", "OFF"))
        all_off_button.pack(side=tk.LEFT)
        self.command_buttons.append(all_off_button)

        log_frame = ttk.LabelFrame(root_frame, text="通信日志")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        self.log_text = tk.Text(log_frame, height=10, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

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

    def parse_port(self) -> int:
        try:
            port = int(self.port_var.get().strip())
        except ValueError as error:
            raise ValueError("端口必须是数字") from error

        if not 1 <= port <= 65535:
            raise ValueError("端口范围必须是 1-65535")

        return port

    def close_connection_locked(self) -> None:
        if self.socket_reader is not None:
            self.socket_reader.close()
            self.socket_reader = None

        if self.control_socket is not None:
            self.control_socket.close()
            self.control_socket = None

    def disconnect(self) -> None:
        self.monitoring_enabled = False
        self.last_heartbeat_ok = None
        self.heartbeat_fail_count = 0
        self.connection_var.set("UDP 已断开")

        if not self.socket_lock.acquire(blocking=False):
            self.append_log("已停止 UDP 自动重连，当前通信结束后会自动释放")
            return

        try:
            self.close_connection_locked()
        finally:
            self.socket_lock.release()

        self.append_log("已停止 UDP 自动重连")

    def parse_response(self, response: str) -> object:
        response = response.strip()
        if response.startswith("{"):
            return json.loads(response)

        return response

    def send_udp_command_once(self, command: str) -> object:
        host = self.host_var.get().strip()
        port = self.parse_port()

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as control_socket:
            control_socket.settimeout(REQUEST_TIMEOUT_SECONDS)
            payload = (command + "\n").encode("utf-8")
            for index in range(UDP_REQUEST_BURST_COUNT):
                control_socket.sendto(payload, (host, port))
                if index + 1 < UDP_REQUEST_BURST_COUNT:
                    time.sleep(UDP_REQUEST_BURST_GAP_SECONDS)

            response_data, _ = control_socket.recvfrom(UDP_RESPONSE_SIZE)

        return self.parse_response(response_data.decode("utf-8", errors="replace"))

    def send_udp_command_with_retries(self, command: str, retry_count: int) -> object:
        self.close_connection_locked()
        last_error: Exception | None = None

        for _ in range(retry_count):
            try:
                return self.send_udp_command_once(command)
            except (OSError, TimeoutError) as error:
                last_error = error

        raise TimeoutError(f"UDP 超时，已重试 {retry_count} 次：{last_error}")

    def send_udp_command(self, command: str) -> object:
        with self.socket_lock:
            return self.send_udp_command_with_retries(command, UDP_RETRY_COUNT)

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
        self.run_worker("连接测试", lambda: self.send_udp_command("PING"))

    def refresh_status(self) -> None:
        self.enable_monitoring()
        self.run_worker("刷新状态", lambda: self.send_udp_command("STATUS"))

    def send_command(self, target: str, action: str) -> None:
        self.enable_monitoring()
        self.run_worker(f"{target} {action}", lambda: self.send_udp_command(f"{target} {action}"))

    def update_channel_status(self, status: dict[str, object]) -> None:
        for channel in CHANNELS:
            if channel in status:
                self.channel_vars[channel].set("接通" if status[channel] else "断开")

    def handle_success(self, task_name: str, result: object) -> None:
        self.last_heartbeat_ok = True
        self.heartbeat_fail_count = 0
        self.connection_var.set("UDP 通信正常")
        self.append_log(f"{task_name}: {result}")

        if isinstance(result, dict):
            self.update_channel_status(result)
            return

        if isinstance(result, str):
            self.update_channel_status_from_line(result)

    def handle_error(self, task_name: str, error: str) -> None:
        self.last_heartbeat_ok = False
        self.heartbeat_fail_count = HEARTBEAT_FAIL_THRESHOLD
        self.connection_var.set("UDP 通信失败")
        self.append_log(f"{task_name}: 失败：{error}")

    def handle_heartbeat_success(self) -> None:
        if not self.monitoring_enabled:
            return

        if self.last_heartbeat_ok is False:
            self.append_log("自动重连: UDP 已恢复")

        self.last_heartbeat_ok = True
        self.heartbeat_fail_count = 0
        self.connection_var.set("UDP 通信正常")

    def handle_heartbeat_error(self, error: str) -> None:
        if not self.monitoring_enabled:
            return

        self.heartbeat_fail_count += 1
        if self.heartbeat_fail_count < HEARTBEAT_FAIL_THRESHOLD:
            return

        self.connection_var.set("UDP 断流，自动重试中")
        now = time.monotonic()

        if (
            self.last_heartbeat_ok is not False
            or now - self.last_heartbeat_error_log_time >= HEARTBEAT_ERROR_LOG_INTERVAL_SECONDS
        ):
            self.append_log(f"自动重连: UDP 暂时无响应，继续重试：{error}")
            self.last_heartbeat_error_log_time = now

        self.last_heartbeat_ok = False

    def update_channel_status_from_line(self, line: str) -> None:
        parts = line.strip().split()
        if len(parts) < 3 or parts[0] != "OK":
            return

        target = parts[1].upper()
        action = parts[2].upper()
        if action not in {"ON", "OFF"}:
            return

        label = "接通" if action == "ON" else "断开"

        if target == "ALL":
            for channel in CHANNELS:
                self.channel_vars[channel].set(label)
            return

        if target in CHANNELS:
            self.channel_vars[target].set(label)

    def _poll_worker_messages(self) -> None:
        while True:
            try:
                message_type, payload = self.worker_messages.get_nowait()
            except queue.Empty:
                break

            task_name, data = payload
            if message_type == "success":
                self.handle_success(str(task_name), data)
            elif message_type == "error":
                self.handle_error(str(task_name), str(data))
            elif message_type == "heartbeat_success":
                self.heartbeat_busy = False
                self.handle_heartbeat_success()
                continue
            elif message_type == "heartbeat_error":
                self.heartbeat_busy = False
                self.handle_heartbeat_error(str(data))
                continue

            self.set_command_busy(False)

        self.root.after(50, self._poll_worker_messages)

    def _schedule_heartbeat(self) -> None:
        self.root.after(HEARTBEAT_INTERVAL_MS, self._run_heartbeat)

    def _run_heartbeat(self) -> None:
        if (
            self.monitoring_enabled
            and not self.command_busy
            and not self.heartbeat_busy
            and self.socket_lock.acquire(blocking=False)
        ):
            self.heartbeat_busy = True

            def wrapped() -> None:
                try:
                    self.send_udp_command_with_retries("PING", HEARTBEAT_RETRY_COUNT)
                    self.worker_messages.put(("heartbeat_success", ("自动心跳", None)))
                except Exception as error:
                    self.worker_messages.put(("heartbeat_error", ("自动心跳", str(error))))
                finally:
                    self.socket_lock.release()

            threading.Thread(target=wrapped, daemon=True).start()

        self._schedule_heartbeat()

    def append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)


def main() -> None:
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


if __name__ == "__main__":
    main()
