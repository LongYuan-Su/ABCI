# -*- coding: utf-8 -*-
"""
吞咽评估范式 — 基于 brainstim 框架的标准化吞咽功能评估实验。

提供完整的吞咽功能评估实验流程：
  实验开始 -> 静息1(5s) -> 想象吞咽1(5s) -> 静息2(5s)
  -> 含水1 -> 温水吞咽1(15s) -> 实验结束

特性：
  - 双重显示后端：PsychoPy (优选) 全屏 / tkinter 全屏回退
  - 四层音频后端：Windows MCI -> pygame -> PsychoPy Sound -> 无声回退
  - LSL Marker 输出（字符串标签，支持 UDP 中继）
  - CSV 日志记录
  - 命令行参数化控制
  - 手动确认步骤（含水后按空格键继续）
  - ESC 键随时中止实验

依赖（均为可选，按优先级回退）：
  psychopy, tkinter, pygame, pylsl

属于 metabci.brainstim 包（原 apps/swallow_paradigm.py）。

用法：
  python metabci/brainstim/paradigm_swallow.py --patient-id P001
  python -m metabci.brainstim.paradigm_swallow --patient-id P001 --epoch-count 2 --debug
"""

# ---- 必须在所有其他 import 之前：压制第三方库警告 ----
import warnings
import logging
import os as _os

warnings.filterwarnings("ignore")
logging.getLogger("psychopy").setLevel(logging.ERROR)
logging.getLogger("pylsl").setLevel(logging.WARNING)
_os.environ.setdefault("LSL_LOG_LEVEL", "WARNING")
_os.environ.setdefault("PSYCHOPY_PARALLEL_PORT", "0")

import os
import sys
import time
import csv
import socket
import argparse
import threading
import datetime
from pathlib import Path
from typing import Optional

# ---- 解压即用: 将项目根目录加入 Python 搜索路径 (必须在其他 metabci import 之前) ----
_CURR_DIR = os.path.abspath(os.path.dirname(__file__))
_METABCI_DIR = os.path.abspath(os.path.join(_CURR_DIR, ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_METABCI_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _METABCI_DIR not in sys.path:
    sys.path.insert(0, _METABCI_DIR)

# ---- MetaBCI framework integration ----
from metabci.brainstim.framework import Experiment


# ===========================================================================
# 自定义异常
# ===========================================================================

class ExperimentAborted(Exception):
    """用户通过 ESC 键中止实验时抛出。"""
    pass


# ===========================================================================
# LSL Marker 发送器
# ===========================================================================

class LSLMarkerSender:
    """LSL 字符串 Marker 输出 + 可选的 UDP 中继。

    通过 pylsl StreamOutlet 发送字符串标记，同时可选地将标记
    以 "timestamp\\tlabel" 格式转发到指定 UDP 地址。

    Parameters
    ----------
    stream_name : str
        LSL Marker 流名称，默认 "swallow_paradigm_markers".
    udp_host : str or None
        UDP 中继目标主机，None 表示不启用 UDP 中继。
    udp_port : int or None
        UDP 中继目标端口，None 表示不启用 UDP 中继。
    """

    def __init__(
        self,
        stream_name: str = "swallow_paradigm_markers",
        udp_host: Optional[str] = None,
        udp_port: Optional[int] = None,
    ):
        self.stream_name = stream_name
        self.udp_host = udp_host
        self.udp_port = udp_port

        # LSL 输出
        self._outlet = None
        self._has_lsl = False
        try:
            import pylsl
            info = pylsl.StreamInfo(
                stream_name, "Markers", 1,
                pylsl.IRREGULAR_RATE, pylsl.cf_string,
                f"swallow_paradigm_{os.getpid()}"
            )
            self._outlet = pylsl.StreamOutlet(info)
            self._has_lsl = True
            print(f"[LSL] Marker 流已创建: '{stream_name}'")
        except ImportError:
            print("[LSL] pylsl 未安装，LSL 输出已禁用")
        except Exception as e:
            print(f"[LSL] 初始化失败: {e}")

        # UDP 中继
        self._udp_sock = None
        self._has_udp = False
        if udp_host and udp_port:
            try:
                self._udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._has_udp = True
                print(f"[UDP] 中继已启用: {udp_host}:{udp_port}")
            except Exception as e:
                print(f"[UDP] 初始化失败: {e}")

    def send(self, label: str) -> None:
        """发送一个字符串标记到 LSL 流和 UDP 中继。

        Parameters
        ----------
        label : str
            标记字符串，如 "实验开始"、"静息1" 等。
        """
        timestamp = time.time()
        # LSL 输出
        if self._has_lsl and self._outlet is not None:
            try:
                self._outlet.push_sample([label])
            except Exception as e:
                print(f"[LSL] 发送失败: {e}")

        # UDP 中继
        if self._has_udp and self._udp_sock is not None:
            try:
                msg = f"{timestamp}\t{label}"
                self._udp_sock.sendto(msg.encode("utf-8"),
                                       (self.udp_host, self.udp_port))
            except Exception as e:
                print(f"[UDP] 发送失败: {e}")

    def close(self) -> None:
        """释放资源。"""
        if self._udp_sock is not None:
            try:
                self._udp_sock.close()
            except Exception:
                pass
            self._udp_sock = None


# ===========================================================================
# 音频播放器 — 多级回退链
# ===========================================================================

class NullAudioPlayer:
    """无声回退播放器——不播放任何声音，仅打印日志。

    当所有其他音频后端均不可用时使用。
    """

    def __init__(self, assets_dir: str = ""):
        self.assets_dir = assets_dir

    def play(self, filename: str, wait: bool = False) -> None:
        """不播放声音。

        Parameters
        ----------
        filename : str
            音频文件名（仅用于日志）。
        wait : bool
            是否等待播放完成（忽略）。
        """
        print(f"[Audio-Null] 静默: {filename}")

    def is_available(self) -> bool:
        """始终返回 True（作为最后回退）。"""
        return True

    def close(self) -> None:
        """释放资源（无操作）。"""
        pass


class MCIAudioPlayer:
    """Windows MCI 音频播放器（回退：os.startfile）。

    优先使用 MCI 播放 MP3 文件；QProcess 子进程中 MCI 不可用时，
    回退到 os.startfile() 调用系统默认播放器。
    """

    def __init__(self, assets_dir: str = ""):
        self.assets_dir = assets_dir
        self._alias_counter = 0
        self._lock = threading.Lock()
        self._mci_available = False
        self._use_startfile = False

        try:
            import ctypes
            self._winmm = ctypes.windll.winmm
            self._winmm.mciSendStringW.argtypes = [
                ctypes.c_wchar_p, ctypes.c_wchar_p,
                ctypes.c_uint, ctypes.c_void_p]
            self._winmm.mciSendStringW.restype = ctypes.c_uint
            buf = ctypes.create_unicode_buffer(256)
            ret = self._winmm.mciSendStringW(
                "capability waveaudio can play", buf, 256, None)
            self._mci_available = (ret == 0)
        except Exception:
            self._winmm = None

        # If MCI is not available, fall back to os.startfile
        if not self._mci_available:
            self._use_startfile = True

    def is_available(self) -> bool:
        return self._mci_available or self._use_startfile

    def play(self, filename: str, wait: bool = False) -> None:
        filepath = self._resolve_path(filename)
        if not os.path.isfile(filepath):
            print(f"[Audio] 文件不存在: {filepath}", flush=True)
            return

        # Signal GUI to play audio (MCI only works in main process with desktop)
        basename = os.path.basename(filename) if not os.path.isabs(filename) else os.path.basename(filepath)
        print(f"AUDIO:{basename}", flush=True)

    def _play_via_startfile(self, filepath: str, wait: bool) -> None:
        """使用 os.startfile 播放（调用 Windows 默认播放器）。"""
        try:
            os.startfile(filepath)
            if wait:
                # 估算播放时间：MP3 约 2-3 秒
                time.sleep(3.0)
        except Exception as e:
            print(f"[Audio] startfile 失败: {e}", flush=True)

    def _close_after_delay(self, alias: str, delay: float) -> None:
        time.sleep(delay)
        try:
            self._winmm.mciSendStringW(f"close {alias}", None, 0, None)
        except Exception:
            pass

    def _resolve_path(self, filename: str) -> str:
        if os.path.isabs(filename):
            return filename
        return os.path.join(self.assets_dir, filename)

    def close(self) -> None:
        pass


class PygameAudioPlayer:
    """pygame 音频播放器。

    使用 pygame.mixer 模块播放音频文件，跨平台支持良好。

    Parameters
    ----------
    assets_dir : str
        音频素材目录路径。
    """

    def __init__(self, assets_dir: str = ""):
        self.assets_dir = assets_dir
        self._initialized = False
        try:
            import pygame
            pygame.mixer.init()
            self._initialized = True
            print("[Audio-pygame] 初始化成功")
        except ImportError:
            print("[Audio-pygame] pygame 未安装")
        except Exception as e:
            print(f"[Audio-pygame] 初始化失败: {e}")

    def is_available(self) -> bool:
        return self._initialized

    def play(self, filename: str, wait: bool = False) -> None:
        """使用 pygame 播放音频文件。

        Parameters
        ----------
        filename : str
            音频文件名（相对于 assets_dir 或绝对路径）。
        wait : bool
            是否阻塞等待播放完成。
        """
        if not self._initialized:
            raise RuntimeError("pygame mixer 未初始化")

        filepath = self._resolve_path(filename)
        if not os.path.isfile(filepath):
            print(f"[Audio-pygame] 文件不存在: {filepath}")
            return

        try:
            import pygame
            sound = pygame.mixer.Sound(filepath)
            channel = sound.play()
            if wait and channel is not None:
                # 等待播放完成
                while channel.get_busy():
                    time.sleep(0.01)
        except Exception as e:
            print(f"[Audio-pygame] 播放失败: {filepath} ({e})")

    def _resolve_path(self, filename: str) -> str:
        if os.path.isabs(filename):
            return filename
        return os.path.join(self.assets_dir, filename)

    def close(self) -> None:
        """释放资源。"""
        if self._initialized:
            try:
                import pygame
                pygame.mixer.quit()
            except Exception:
                pass


class PsychoPyAudioPlayer:
    """PsychoPy 音频播放器。

    使用 PsychoPy 的 sound.Sound 模块播放音频，与 PsychoPy 显示后端共享依赖。

    Parameters
    ----------
    assets_dir : str
        音频素材目录路径。
    """

    def __init__(self, assets_dir: str = ""):
        self.assets_dir = assets_dir
        self._initialized = False
        try:
            from psychopy import sound
            sound.init()
            self._initialized = True
            print("[Audio-PsychoPy] 初始化成功")
        except ImportError:
            print("[Audio-PsychoPy] psychopy 未安装")
        except Exception as e:
            print(f"[Audio-PsychoPy] 初始化失败: {e}")

    def is_available(self) -> bool:
        return self._initialized

    def play(self, filename: str, wait: bool = False) -> None:
        """使用 PsychoPy Sound 播放音频文件。

        Parameters
        ----------
        filename : str
            音频文件名（相对于 assets_dir 或绝对路径）。
        wait : bool
            是否阻塞等待播放完成。
        """
        if not self._initialized:
            raise RuntimeError("PsychoPy Sound 未初始化")

        filepath = self._resolve_path(filename)
        if not os.path.isfile(filepath):
            print(f"[Audio-PsychoPy] 文件不存在: {filepath}")
            return

        try:
            from psychopy import sound
            s = sound.Sound(filepath)
            s.play()
            if wait:
                # 等待播放完成（估算时长）
                time.sleep(s.getDuration())
        except Exception as e:
            print(f"[Audio-PsychoPy] 播放失败: {filepath} ({e})")

    def _resolve_path(self, filename: str) -> str:
        if os.path.isabs(filename):
            return filename
        return os.path.join(self.assets_dir, filename)

    def close(self) -> None:
        """释放资源。"""
        pass


def create_audio_player(assets_dir: str):
    """音频播放器工厂函数 —— 按优先级回退。

    回退链: Windows MCI -> pygame -> PsychoPy Sound -> 无声回退

    Parameters
    ----------
    assets_dir : str
        音频素材目录路径。

    Returns
    -------
    player
        第一个可用的音频播放器实例。
    """
    # 1) pygame — 跨平台，最可靠
    pg = PygameAudioPlayer(assets_dir)
    if pg.is_available():
        print("[Audio] 使用 pygame 后端", flush=True)
        return pg

    # 2) Windows MCI（仅 Windows）
    mci = MCIAudioPlayer(assets_dir)
    if mci.is_available():
        print("[Audio] 使用 Windows MCI 后端", flush=True)
        return mci

    # 3) PsychoPy Sound
    pp = PsychoPyAudioPlayer(assets_dir)
    if pp.is_available():
        print("[Audio] 使用 PsychoPy 后端", flush=True)
        return pp

    # 4) 无声回退
    print("[Audio] 所有音频后端不可用，使用静默回退", flush=True)
    return NullAudioPlayer(assets_dir)


# ===========================================================================
# 显示后端
# ===========================================================================

class PsychoPyBackend:
    """PsychoPy 全屏显示后端。

    使用 PsychoPy 的 Window + TextStim 实现全屏黑色背景白字显示，
    支持标题、副标题和页脚三行文本。

    Parameters
    ----------
    debug : bool
        调试模式：windowed (1024x768) 而非 fullscr。
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._win = None
        self._title_stim = None
        self._subtitle_stim = None
        self._footer_stim = None
        self._initialized = False

        try:
            from psychopy import visual, event, core
            self._psychopy_visual = visual
            self._psychopy_event = event
            self._psychopy_core = core

            # 中文字体回退链
            _font = "Microsoft YaHei"

            self._win = visual.Window(
                size=(1024, 768) if debug else None,
                fullscr=not debug,
                screen=0,
                winType="pyglet",
                allowGUI=False,
                allowStencil=False,
                color="black",
                units="pix",
                waitBlanking=not debug,  # 全屏 vsync 开，窗口模式关
            )
            win_size = self._win.size

            # 标题：屏幕上方 1/3 处
            self._title_stim = visual.TextStim(
                self._win,
                text="",
                pos=(0, win_size[1] * 0.2),
                height=48,
                color="white",
                bold=True,
                font=_font,
                wrapWidth=win_size[0] * 0.85,
            )
            # 副标题：屏幕中央
            self._subtitle_stim = visual.TextStim(
                self._win,
                text="",
                pos=(0, 0),
                height=36,
                color="white",
                font=_font,
                wrapWidth=win_size[0] * 0.85,
            )
            # 页脚：屏幕下方
            self._footer_stim = visual.TextStim(
                self._win,
                text="",
                pos=(0, -win_size[1] * 0.25),
                height=24,
                color="#AAAAAA",
                font=_font,
                wrapWidth=win_size[0] * 0.85,
            )
            self._initialized = True
            print("[Display-PsychoPy] 全屏显示初始化成功")
        except ImportError:
            print("[Display-PsychoPy] psychopy 未安装")
        except Exception as e:
            print(f"[Display-PsychoPy] 初始化失败: {e}")

    def is_available(self) -> bool:
        return self._initialized

    def show_message(self, title: str = "", subtitle: str = "",
                     footer: str = "") -> None:
        """更新屏幕显示。

        Parameters
        ----------
        title : str
            主标题文本（大号，上方）。
        subtitle : str
            副标题文本（中号，中央）。
        footer : str
            页脚文本（小号，下方），通常显示按键提示。
        """
        if not self._initialized:
            return
        try:
            self._title_stim.setText(title)
            self._subtitle_stim.setText(subtitle)
            self._footer_stim.setText(footer)
            self._title_stim.draw()
            self._subtitle_stim.draw()
            self._footer_stim.draw()
            self._win.flip()
        except Exception as e:
            print(f"[Display-PsychoPy] 显示更新失败: {e}")

    def poll_key(self) -> Optional[str]:
        """非阻塞轮询按键。

        Returns
        -------
        key : str or None
            按键名称（小写），如 "space", "escape"；无按键则返回 None。
        """
        if not self._initialized:
            return None
        try:
            keys = self._psychopy_event.getKeys(keyList=["space", "escape"])
            if keys:
                return keys[0]
        except Exception:
            pass
        return None

    def clear_keys(self) -> None:
        """清空按键缓冲区。"""
        if not self._initialized:
            return
        try:
            self._psychopy_event.clearEvents()
        except Exception:
            pass

    def close(self) -> None:
        """关闭窗口并释放资源。"""
        if self._win is not None:
            try:
                self._win.close()
            except Exception:
                pass
            self._win = None
            self._initialized = False


class TkinterBackend:
    """tkinter 全屏显示后端。

    全屏黑色背景 + 白字，Microsoft YaHei 字体。
    当 PsychoPy 不可用时作为回退。

    Parameters
    ----------
    debug : bool
        调试模式：非全屏固定窗口。
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._root = None
        self._title_label = None
        self._subtitle_label = None
        self._footer_label = None
        self._key_queue = []
        self._initialized = False
        self._lock = threading.Lock()

        try:
            import tkinter as tk
            self._tk = tk
            self._root = tk.Tk()
            self._root.configure(bg="black")
            self._root.title("吞咽评估实验")
            # 允许窗口关闭按钮
            self._root.protocol("WM_DELETE_WINDOW", self._on_close)

            if debug:
                self._root.geometry("1024x768")
            else:
                self._root.attributes("-fullscreen", True)

            # 绑定按键
            self._root.bind("<KeyPress-space>", lambda e: self._on_key("space"))
            self._root.bind("<KeyPress-Escape>", lambda e: self._on_key("escape"))
            self._root.bind("<KeyPress-space>", lambda e: self._on_key("space"))

            # 标题
            self._title_label = tk.Label(
                self._root, text="",
                font=("Microsoft YaHei", 36, "bold"),
                fg="white", bg="black", wraplength=900,
            )
            self._title_label.pack(expand=False, pady=(200, 20))

            # 副标题
            self._subtitle_label = tk.Label(
                self._root, text="",
                font=("Microsoft YaHei", 24),
                fg="white", bg="black", wraplength=900,
            )
            self._subtitle_label.pack(expand=False, pady=20)

            # 页脚
            self._footer_label = tk.Label(
                self._root, text="",
                font=("Microsoft YaHei", 16),
                fg="#AAAAAA", bg="black", wraplength=900,
            )
            self._footer_label.pack(expand=False, pady=20)

            self._root.update()
            self._initialized = True
            print("[Display-tkinter] 全屏显示初始化成功")
        except ImportError:
            print("[Display-tkinter] tkinter 不可用")
        except Exception as e:
            print(f"[Display-tkinter] 初始化失败: {e}")

    def is_available(self) -> bool:
        return self._initialized

    def show_message(self, title: str = "", subtitle: str = "",
                     footer: str = "") -> None:
        """更新屏幕显示。

        Parameters
        ----------
        title : str
            主标题文本。
        subtitle : str
            副标题文本。
        footer : str
            页脚文本。
        """
        if not self._initialized:
            return
        try:
            self._title_label.config(text=title)
            self._subtitle_label.config(text=subtitle)
            self._footer_label.config(text=footer)
            self._root.update()
        except Exception as e:
            print(f"[Display-tkinter] 显示更新失败: {e}")

    def poll_key(self) -> Optional[str]:
        """非阻塞轮询按键。

        Returns
        -------
        key : str or None
        """
        if not self._initialized:
            return None
        try:
            self._root.update()
            with self._lock:
                if self._key_queue:
                    return self._key_queue.pop(0)
        except Exception:
            pass
        return None

    def clear_keys(self) -> None:
        """清空按键缓冲区。"""
        with self._lock:
            self._key_queue.clear()
        try:
            self._root.update()
        except Exception:
            pass

    def close(self) -> None:
        """关闭窗口并释放资源。"""
        if self._root is not None:
            try:
                self._root.destroy()
            except Exception:
                pass
            self._root = None
            self._initialized = False

    def _on_key(self, key: str) -> None:
        """按键回调。"""
        with self._lock:
            self._key_queue.append(key)

    def _on_close(self) -> None:
        """窗口关闭按钮回调——发送 escape 模拟退出。"""
        self._on_key("escape")


def create_display_backend(debug: bool = False):
    """显示后端工厂函数 —— 按优先级回退。

    回退链: PsychoPy 全屏 -> tkinter 全屏

    Parameters
    ----------
    debug : bool
        调试模式。

    Returns
    -------
    backend
        第一个可用的显示后端实例。

    Raises
    ------
    RuntimeError
        如果所有显示后端均不可用。
    """
    # 1) PsychoPy
    pp = PsychoPyBackend(debug=debug)
    if pp.is_available():
        print("[Display] 使用 PsychoPy 后端")
        return pp

    # 2) tkinter
    tk = TkinterBackend(debug=debug)
    if tk.is_available():
        print("[Display] 使用 tkinter 后端")
        return tk

    raise RuntimeError("所有显示后端均不可用。请安装 psychopy 或确认 tkinter 可用。")


# ===========================================================================
# 事件配置
# ===========================================================================

# 声音键 -> 音频文件名的映射
SOUND_FILE_MAP = {
    "实验开始": "实验开始.mp3",
    "静息": "请保持放松.mp3",
    "想象吞咽": "想象吞咽.mp3",
    "含水": "含水.mp3",
    "温水吞咽": "温水吞咽.mp3",
    "实验结束": "实验结束.mp3",
}

# 事件配置列表：每个 epoch 中依次执行的事件
# 每个元素为 dict:
#   name           : str   — 事件名称（同时作为 LSL Marker 标签）
#   delay          : float — 事件显示持续时间（秒）；0 表示仅播放声音后立即继续
#   sound          : str   — 声音键，对应 SOUND_FILE_MAP 中的键
#   manual_confirm : bool  — 是否等待用户按空格键确认
#   subtitle       : str   — 副标题文本
EVENTS = [
    {
        "name": "实验开始",
        "delay": 4.0,
        "sound": "实验开始",
        "manual_confirm": False,
        "subtitle": "准备开始实验",
    },
    {
        "name": "静息1",
        "delay": 5.0,
        "sound": "静息",
        "manual_confirm": False,
        "subtitle": "请保持放松，静息状态",
    },
    {
        "name": "想象吞咽1",
        "delay": 5.0,
        "sound": "想象吞咽",
        "manual_confirm": False,
        "subtitle": "请想象吞咽动作",
    },
    {
        "name": "静息2",
        "delay": 5.0,
        "sound": "静息",
        "manual_confirm": False,
        "subtitle": "请保持放松，静息状态",
    },
    {
        "name": "含水1",
        "delay": 5.0,
        "sound": "含水",
        "manual_confirm": True,
        "subtitle": "请含一口温水，准备好后按空格键",
    },
    {
        "name": "温水吞咽1",
        "delay": 15.0,
        "sound": "温水吞咽",
        "manual_confirm": False,
        "subtitle": "请进行吞咽",
    },
    {
        "name": "实验结束",
        "delay": 4.0,
        "sound": "实验结束",
        "manual_confirm": False,
        "subtitle": "实验已结束，感谢您的配合",
    },
]


# ===========================================================================
# 范式运行主逻辑
# ===========================================================================

def run_paradigm(
    patient_id: str,
    epoch_count: int = 1,
    experiment_date: str = "",
    marker_sender: Optional[LSLMarkerSender] = None,
    display=None,
    audio_player=None,
    log_dir: str = "logs",
    debug: bool = False,
) -> list:
    """运行吞咽评估范式的主逻辑。

    按 epoch 循环执行 EVENTS 列表中的每个事件步骤。
    每个事件的流程：
      1. 发送 LSL Marker
      2. 显示标题/副标题/页脚
      3. 播放声音
      4. 等待 delay 秒（期间可 ESC 退出）
      5. 若 manual_confirm，等待空格键
      6. 记录 CSV 日志
      7. 进入下一个事件

    Parameters
    ----------
    patient_id : str
        患者编号。
    epoch_count : int
        实验 epoch 重复次数，默认 1。
    experiment_date : str
        实验日期 (YYYY-MM-DD)，为空则使用当前日期。
    marker_sender : LSLMarkerSender or None
        LSL/UDP Marker 发送器。
    display
        显示后端实例。
    audio_player
        音频播放器实例。
    log_dir : str
        日志输出目录。
    debug : bool
        调试模式。

    Returns
    -------
    log_records : list of dict
        完整的日志记录列表。
    """
    if experiment_date:
        exp_date = experiment_date
    else:
        exp_date = datetime.date.today().isoformat()

    # 创建日志目录
    log_path = Path(log_dir) / patient_id
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"swallow_paradigm_log_{timestamp_str}.csv"
    csv_filepath = log_path / csv_filename

    print(f"[Log] 日志文件: {csv_filepath}")

    # 打开 CSV 文件用于写入
    csv_file = open(str(csv_filepath), "w", newline="", encoding="utf-8-sig")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "event_index", "event_name",
        "timestamp_sec", "system_time",
    ])

    log_records = []
    event_counter = 0  # 全局事件序号（跨 epoch）

    # 计算预计总时长
    total_duration = 0
    for evt in EVENTS:
        total_duration += evt["delay"]
        if evt["manual_confirm"]:
            total_duration += 30  # 估算手动确认等待时间
    total_duration *= epoch_count
    print(f"[Paradigm] 预计总时长: ~{total_duration:.0f}s ({epoch_count} epoch(s))")

    # --- 主循环：epoch x events ---
    try:
        for epoch in range(1, epoch_count + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{epoch_count}")
            print(f"{'='*50}")

            for evt_idx, evt in enumerate(EVENTS):
                event_counter += 1
                name = evt["name"]
                delay = evt["delay"]
                sound_key = evt["sound"]
                manual = evt["manual_confirm"]
                subtitle = evt["subtitle"]

                epoch_label = name
                if epoch_count > 1:
                    epoch_label = f"E{epoch}_{name}"

                ts = time.time()
                sys_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                # 1) 发送 LSL Marker
                if marker_sender is not None:
                    marker_sender.send(epoch_label)

                print(f"EVENT:{epoch_label}", flush=True)

                if debug:
                    print(f"  [Event {event_counter}] {epoch_label}  "
                          f"delay={delay}s  sound='{sound_key}'  "
                          f"manual={manual}  ts={ts:.3f}")

                # 2) 更新显示
                title = f"吞咽评估实验"
                if epoch_count > 1:
                    title += f" (Epoch {epoch}/{epoch_count})"
                footer = ""
                if manual:
                    footer = "请准备好后按【空格键】继续"
                else:
                    footer = "请保持放松 | 按【ESC】键退出实验"

                display.show_message(
                    title=title,
                    subtitle=subtitle,
                    footer=footer,
                )

                # 3) 播放声音
                audio_file = SOUND_FILE_MAP.get(sound_key, "")
                if audio_file and audio_player.is_available():
                    try:
                        audio_player.play(audio_file, wait=True)
                    except Exception as e:
                        print(f"  [Audio] 播放异常: {e}", flush=True)

                # 4) 等待 delay 秒（非阻塞，检查 ESC 键）
                if delay > 0:
                    delay_start = time.time()
                    while (time.time() - delay_start) < delay:
                        key = display.poll_key()
                        if key == "escape":
                            raise ExperimentAborted("用户按 ESC 退出")
                        time.sleep(0.05)

                # 5) 若需要手动确认，等待空格键
                if manual:
                    display.show_message(
                        title=title,
                        subtitle=subtitle + "\n\n请准备好后按【空格键】继续",
                        footer="按【空格键】确认 | 按【ESC】键退出实验",
                    )
                    display.clear_keys()
                    while True:
                        key = display.poll_key()
                        if key == "space":
                            break
                        elif key == "escape":
                            raise ExperimentAborted("用户按 ESC 退出")
                        time.sleep(0.05)

                # 6) 记录日志
                record = {
                    "epoch": epoch,
                    "event_index": event_counter,
                    "event_name": epoch_label,
                    "timestamp_sec": ts,
                    "system_time": sys_time,
                }
                log_records.append(record)

                csv_writer.writerow([
                    epoch, event_counter, epoch_label,
                    f"{ts:.6f}", sys_time,
                ])
                csv_file.flush()

    except ExperimentAborted:
        print("\n[Paradigm] 实验已被用户中止（ESC键）")

        # 记录中止事件
        ts = time.time()
        sys_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        abort_record = {
            "epoch": -1,
            "event_index": event_counter + 1,
            "event_name": "实验中止",
            "timestamp_sec": ts,
            "system_time": sys_time,
        }
        log_records.append(abort_record)
        csv_writer.writerow([
            -1, event_counter + 1, "实验中止",
            f"{ts:.6f}", sys_time,
        ])

        # 发送中止标记
        if marker_sender is not None:
            marker_sender.send("实验中止")

        # 显示中止信息
        try:
            display.show_message(
                title="实验已中止",
                subtitle="按 ESC 键退出，或等待自动关闭...",
                footer="",
            )
            wait_start = time.time()
            while (time.time() - wait_start) < 3.0:
                key = display.poll_key()
                if key == "escape":
                    break
                time.sleep(0.05)
        except Exception:
            pass

    except KeyboardInterrupt:
        print("\n[Paradigm] 实验已被 Ctrl+C 中断")

    finally:
        csv_file.close()
        print(f"[Log] 日志已保存: {csv_filepath}")

    return log_records


# ===========================================================================
# 命令行入口
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="吞咽评估范式 — 标准化吞咽功能评估实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python metabci/brainstim/paradigm_swallow.py --patient-id P001
  python -m metabci.brainstim.paradigm_swallow --patient-id P001 --epoch-count 2 --debug
  python -m metabci.brainstim.paradigm_swallow --patient-id P001 --marker-stream-name swallow_markers
  python -m metabci.brainstim.paradigm_swallow --patient-id P001 --marker-udp-host 127.0.0.1 --marker-udp-port 9999
  python -m metabci.brainstim.paradigm_swallow --patient-id P001 --experiment-date 2025-06-01 --eeg-marker
        """,
    )

    parser.add_argument(
        "--patient-id", type=str, required=True,
        help="患者编号 (必需)",
    )
    parser.add_argument(
        "--epoch-count", type=int, default=1,
        help="实验 epoch 重复次数 (默认: 1)",
    )
    parser.add_argument(
        "--experiment-date", type=str, default="",
        help="实验日期 YYYY-MM-DD (默认: 当天)",
    )
    parser.add_argument(
        "--eeg-marker", action="store_true",
        help="启用 LSL Marker 输出",
    )
    parser.add_argument(
        "--marker-stream-name", type=str,
        default="swallow_paradigm_markers",
        help="LSL Marker 流名称 (默认: swallow_paradigm_markers)",
    )
    parser.add_argument(
        "--marker-udp-host", type=str, default=None,
        help="UDP Marker 中继目标主机 (不指定则不启用)",
    )
    parser.add_argument(
        "--marker-udp-port", type=int, default=None,
        help="UDP Marker 中继目标端口 (不指定则不启用)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="调试模式 (窗口化、详细日志)",
    )

    return parser


def run_swallow_paradigm_via_experiment(
    patient_id: str = "P001",
    epoch_count: int = 1,
    experiment_date: str = "",
    debug: bool = False,
) -> list:
    """通过 brainstim.framework.Experiment 运行吞咽范式。

    使用 ``Experiment.get_window()`` 管理 PsychoPy 窗口生命周期，
    注册 swallow 范式并通过 ``Experiment.run()`` 启动。

    Parameters
    ----------
    patient_id : str
    epoch_count : int
    experiment_date : str
    debug : bool

    Returns
    -------
    log_records : list
    """
    assets_dir = os.path.join(_CURR_DIR, "assets")
    log_dir = os.path.join(_PROJECT_ROOT, "logs")

    # Initialise components (display, audio, markers — same as standalone)
    display = create_display_backend(debug=debug)
    audio_player = create_audio_player(assets_dir)

    # Build the paradigm runner
    def _run(win=None, **kwargs):
        """Paradigm function compatible with Experiment.register_paradigm()."""
        return run_paradigm(
            patient_id=patient_id,
            epoch_count=epoch_count,
            experiment_date=experiment_date,
            marker_sender=None,
            display=display,
            audio_player=audio_player,
            log_dir=log_dir,
            debug=debug,
        )

    # Register with Experiment framework
    ex = Experiment(
        win_size=(1024, 768),
        screen_id=0,
        is_fullscr=not debug,
        bg_color_warm=[0, 0, 0],
    )
    ex.register_paradigm("swallow_assessment", _run)
    try:
        ex.initEvent()
    except RuntimeError:
        pass  # ESC key already registered from previous run

    # Show start screen, then run
    display.show_message(
        title="吞咽评估实验",
        subtitle="实验即将开始，请保持放松\n\n按【空格键】开始实验",
        footer="按【ESC】键退出 | 按【空格键】开始",
    )
    display.clear_keys()

    while True:
        key = display.poll_key()
        if key == "space":
            break
        elif key == "escape":
            display.close()
            if audio_player:
                audio_player.close()
            return []
        time.sleep(0.05)

    # Run via Experiment
    ex.run()

    # Cleanup
    ex.closeEvent()
    display.close()
    if audio_player:
        audio_player.close()

    # Collect results from the paradigm's CSV log
    log_records = []
    csv_path = os.path.join(
        log_dir, patient_id,
        f"swallow_paradigm_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    if os.path.isfile(csv_path):
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            log_records = list(csv.DictReader(f))

    return log_records


def _find_latest_log_path(log_dir: str, patient_id: str) -> Optional[str]:
    """Return the newest swallow paradigm CSV log for a patient."""
    patient_log_dir = Path(log_dir) / patient_id
    if not patient_log_dir.is_dir():
        return None
    csv_files = sorted(
        patient_log_dir.glob("swallow_paradigm_log_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(csv_files[0]) if csv_files else None

def main():
    """主入口：解析参数、初始化组件、运行范式、清理资源。"""
    parser = build_parser()
    args = parser.parse_args()

    # --- 参数校验 ---
    if args.epoch_count < 1:
        print("错误: --epoch-count 必须 >= 1")
        sys.exit(1)

    # 确定 assets 目录（素材目录，与此文件同目录）
    assets_dir = os.path.join(_CURR_DIR, "assets")
    if not os.path.isdir(assets_dir):
        print(f"警告: 素材目录不存在: {assets_dir}")
        print("音频播放将使用静默回退。")

    log_dir = os.path.join(_PROJECT_ROOT, "logs")

    # --- 打印实验配置 ---
    print("=" * 55)
    print("  吞咽评估范式 — Swallow Assessment Paradigm")
    print("=" * 55)
    print(f"  患者编号:     {args.patient_id}")
    print(f"  Epoch 次数:   {args.epoch_count}")
    print(f"  实验日期:     {args.experiment_date or '今天'}")
    print(f"  LSL Marker:   {'开启' if args.eeg_marker else '关闭'}")
    if args.eeg_marker:
        print(f"  Marker 流名:  {args.marker_stream_name}")
    if args.marker_udp_host and args.marker_udp_port:
        print(f"  UDP 中继:     {args.marker_udp_host}:{args.marker_udp_port}")
    print(f"  调试模式:     {'开启' if args.debug else '关闭'}")
    print(f"  素材目录:     {assets_dir}")
    print(f"  日志目录:     {log_dir}")
    print("=" * 55)

    # --- 初始化组件 ---
    # 显示后端
    display = create_display_backend(debug=args.debug)

    # 音频播放器
    audio_player = create_audio_player(assets_dir)

    # LSL Marker 发送器
    marker_sender = None
    if args.eeg_marker:
        marker_sender = LSLMarkerSender(
            stream_name=args.marker_stream_name,
            udp_host=args.marker_udp_host,
            udp_port=args.marker_udp_port,
        )

    # --- 显示开始画面 ---
    display.show_message(
        title="吞咽评估实验",
        subtitle="实验即将开始，请保持放松\n\n按【空格键】开始实验",
        footer="按【ESC】键退出 | 按【空格键】开始",
    )
    display.clear_keys()

    # 等待开始或退出
    while True:
        key = display.poll_key()
        if key == "space":
            break
        elif key == "escape":
            print("用户在开始前退出。")
            display.close()
            if audio_player:
                audio_player.close()
            if marker_sender:
                marker_sender.close()
            return
        time.sleep(0.05)

    # --- 运行范式 ---
    log_records = run_paradigm(
        patient_id=args.patient_id,
        epoch_count=args.epoch_count,
        experiment_date=args.experiment_date,
        marker_sender=marker_sender,
        display=display,
        audio_player=audio_player,
        log_dir=log_dir,
        debug=args.debug,
    )


    latest_csv = _find_latest_log_path(log_dir, args.patient_id)
    if latest_csv and not any(r["event_name"] == "实验中止" for r in log_records):
        try:
            from metabci.brainflow.processing.assessment import assess_from_paradigm_log

            model_path = os.path.join(
                _PROJECT_ROOT, "models", "final_quantification_model.pt")
            report = assess_from_paradigm_log(
                patient_id=args.patient_id,
                csv_log_path=latest_csv,
                model_path=model_path if os.path.isfile(model_path) else None,
            )
            print(
                "[Assessment] 综合评分: "
                f"{report['composite_score']}/100 "
                f"({report['composite_level']})"
            )
        except Exception as e:
            print(f"[Assessment] 评估失败: {e}")
    # --- 显示结束画面 ---
    display.show_message(
        title="实验完成",
        subtitle=f"实验已全部结束，感谢您的配合！\n\n"
                f"共完成 {len([r for r in log_records if r['event_name'] != '实验中止'])} 个事件",
        footer="请等待工作人员引导 | 按【ESC】键关闭",
    )
    display.clear_keys()
    wait_start = time.time()
    while (time.time() - wait_start) < 10.0:
        key = display.poll_key()
        if key == "escape":
            break
        time.sleep(0.05)

    # --- 清理资源 ---
    print("\n[Cleanup] 正在释放资源...")
    if marker_sender is not None:
        marker_sender.close()
    display.close()
    if audio_player is not None:
        audio_player.close()

    print("[Cleanup] 完成。实验结束。")
    print(f"[Stats] 总事件数: {len(log_records)}")
    aborted = any(r["event_name"] == "实验中止" for r in log_records)
    if aborted:
        print("[Stats] 状态: 实验被中止")
    else:
        print("[Stats] 状态: 实验正常完成")


# ===========================================================================
# 模块入口
# ===========================================================================

if __name__ == "__main__":
    main()
