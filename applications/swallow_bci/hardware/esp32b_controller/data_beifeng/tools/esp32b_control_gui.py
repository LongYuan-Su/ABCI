import json
import queue
import socket
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk


DEFAULT_HOST = "192.168.4.1"
DEFAULT_PORT = 3333
REQUEST_TIMEOUT_SECONDS = 2.0
CHANNELS = ("CH1", "CH2", "CH3", "CH4")


class Esp32ControlApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ESP32B WiFi 控制上位机")
        self.root.geometry("640x470")
        self.root.minsize(580, 430)

        self.host_var = tk.StringVar(value=DEFAULT_HOST)
        self.port_var = tk.StringVar(value=str(DEFAULT_PORT))
        self.connection_var = tk.StringVar(value="TCP 未连接")
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

        self._build_ui()
        self._poll_worker_messages()

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

    def parse_port(self) -> int:
        try:
            port = int(self.port_var.get().strip())
        except ValueError as error:
            raise ValueError("端口必须是数字") from error

        if not 1 <= port <= 65535:
            raise ValueError("端口范围必须是 1-65535")

        return port

    def ensure_connection_locked(self) -> None:
        if self.control_socket is not None:
            return

        host = self.host_var.get().strip()
        port = self.parse_port()
        control_socket = socket.create_connection((host, port), timeout=REQUEST_TIMEOUT_SECONDS)
        control_socket.settimeout(REQUEST_TIMEOUT_SECONDS)
        control_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self.control_socket = control_socket
        self.socket_reader = control_socket.makefile("r", encoding="utf-8", newline="\n")

        greeting = self.read_line_locked()
        if not greeting.startswith("OK"):
            self.close_connection_locked()
            raise ConnectionError(greeting or "ESP32B 没有返回握手信息")

    def close_connection_locked(self) -> None:
        if self.socket_reader is not None:
            self.socket_reader.close()
            self.socket_reader = None

        if self.control_socket is not None:
            self.control_socket.close()
            self.control_socket = None

    def disconnect(self) -> None:
        if not self.socket_lock.acquire(blocking=False):
            self.append_log("当前正在通信，等这条命令完成后再断开")
            return

        try:
            self.close_connection_locked()
        finally:
            self.socket_lock.release()

        self.connection_var.set("TCP 未连接")
        self.append_log("已断开 TCP 连接")

    def read_line_locked(self) -> str:
        if self.socket_reader is None:
            raise ConnectionError("TCP 未连接")

        response = self.socket_reader.readline()
        if response == "":
            self.close_connection_locked()
            raise ConnectionError("ESP32B 已断开连接")

        return response.strip()

    def send_tcp_command_once_locked(self, command: str) -> object:
        self.ensure_connection_locked()

        if self.control_socket is None:
            raise ConnectionError("TCP 未连接")

        self.control_socket.sendall((command + "\n").encode("utf-8"))
        response = self.read_line_locked()

        if response.startswith("{"):
            return json.loads(response)

        return response

    def send_tcp_command(self, command: str) -> object:
        with self.socket_lock:
            try:
                return self.send_tcp_command_once_locked(command)
            except Exception:
                self.close_connection_locked()

            self.ensure_connection_locked()
            return self.send_tcp_command_once_locked(command)

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
        self.run_worker("连接测试", lambda: self.send_tcp_command("PING"))

    def refresh_status(self) -> None:
        self.run_worker("刷新状态", lambda: self.send_tcp_command("STATUS"))

    def send_command(self, target: str, action: str) -> None:
        self.run_worker(f"{target} {action}", lambda: self.send_tcp_command(f"{target} {action}"))

    def update_channel_status(self, status: dict[str, object]) -> None:
        for channel in CHANNELS:
            if channel in status:
                self.channel_vars[channel].set("接通" if status[channel] else "断开")

    def handle_success(self, task_name: str, result: object) -> None:
        self.connection_var.set("TCP 连接正常")
        self.append_log(f"{task_name}: {result}")

        if isinstance(result, dict):
            self.update_channel_status(result)
            return

        if isinstance(result, str):
            self.update_channel_status_from_line(result)

    def handle_error(self, task_name: str, error: str) -> None:
        self.connection_var.set("TCP 连接失败")
        self.append_log(f"{task_name}: 失败：{error}")

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
            else:
                self.handle_error(str(task_name), str(data))
            self.set_command_busy(False)

        self.root.after(50, self._poll_worker_messages)

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
