# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Optional
"""Extended data sources for MetaBCI brainflow.

Provides amplifier-like components referenced by apps but not in base amplifiers.py:
  - RealTimeBuffer: ring buffer with event tracking
  - SimSwallowAmplifier: simulated swallow EEG amplifier
  - OpenBCISource: LSL-based data source
  - WiFiShieldAmplifier: TCP/IP WiFi connection to OpenBCI
  - create_source: factory function

These extend the base amplifiers module without modifying existing code.
"""

import time
import json
import socket
import struct
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
import numpy as np

from ..amplifiers import RingBuffer, LSLInlet
from ..logger import get_logger
logger = get_logger("sources")


class RealTimeBuffer:
    """Real-time data buffer with event tracking.

    Wraps RingBuffer for multi-channel data with marker events.

    Parameters
    ----------
    n_channels : int
        Number of data channels.
    max_size : int
        Maximum buffer size in samples.
    """

    def __init__(self, n_channels: int = 8, max_size: int = 7500):
        self.n_channels = n_channels
        self.max_size = max_size
        self._buffer = RingBuffer(size=max_size)
        self._events = RingBuffer(size=max_size)

    def push(self, data: np.ndarray, events: list = None):
        """Push a data chunk (n_channels, n_samples) with optional events."""
        n_samples = data.shape[1] if data.ndim > 1 else 1
        for i in range(n_samples):
            sample = data[:, i] if data.ndim > 1 else data
            self._buffer.append(sample)
        if events:
            for ev in events:
                self._events.append(ev)

    def get_recent(self, n_samples: int) -> Optional[np.ndarray]:
        """Get the most recent n_samples as (n_channels, n_samples)."""
        if len(self._buffer) == 0:
            return None
        n = min(n_samples, len(self._buffer))
        all_data = list(self._buffer)
        recent = np.array(all_data[-n:], dtype=np.float32).T  # (n_ch, n)
        return recent

    def get_all(self) -> np.ndarray:
        """Get all data as (n_channels, n_samples)."""
        if len(self._buffer) == 0:
            return np.array([])
        return np.array(list(self._buffer), dtype=np.float32).T

    def clear(self):
        """Clear the buffer."""
        self._buffer.clear()
        self._events.clear()

    @property
    def n_samples(self) -> int:
        return len(self._buffer)


class SimSwallowAmplifier:
    """Simulated swallow EEG amplifier.

    Generates synthetic multi-channel data with P300 and swallow events,
    mimicking a real EEG amplifier's streaming interface.

    Parameters
    ----------
    srate : float
        Sampling rate in Hz.
    n_channels : int
        Number of EEG channels.
    duration : float
        Total duration in seconds.
    chunk_size : int
        Samples per chunk.
    """

    def __init__(self, srate: float = 250.0, n_channels: int = 8,
                 duration: float = 60.0, chunk_size: int = 125):
        from .simulator import SwallowDataSimulator

        self.srate = srate
        self.n_channels = n_channels
        self.duration = duration
        self.chunk_size = chunk_size
        self._simulator = SwallowDataSimulator(
            srate=srate, duration=duration, n_channels=n_channels)
        self._data = None
        self._events = []
        self._pos = 0

    def generate(self) -> dict:
        """Generate all data at once.

        Returns
        -------
        dataset : dict with 'data' (n_channels, n_samples) and 'events'
        """
        n_events = max(1, int(self.duration / 5))
        p300_times = np.linspace(5, self.duration - 5, n_events)
        dataset = self._simulator.generate(p300_times=p300_times)
        # Return only EEG channels
        dataset["data"] = dataset["data"][:self.n_channels, :]
        self._data = dataset["data"]
        self._events = sorted(dataset["events"], key=lambda x: x[0])
        self._pos = 0
        return dataset

    def read_chunk(self) -> Optional[np.ndarray]:
        """Read the next chunk of data (n_channels+3, chunk_size).

        Returns None when data is exhausted.
        """
        if self._data is None or self._pos >= self._data.shape[1]:
            return None
        end = min(self._pos + self.chunk_size, self._data.shape[1])
        chunk = self._data[:, self._pos:end]
        self._pos = end
        return chunk

    def get_recent_events(self, start_sample: int, end_sample: int) -> list:
        """Get events within [start_sample, end_sample)."""
        return [(s - start_sample, l) for s, l in self._events
                if start_sample <= s < end_sample]

    def is_streaming(self) -> bool:
        return self._data is not None and self._pos < self._data.shape[1]

    def get_srate(self) -> float:
        return self.srate

    def get_n_channels(self) -> int:
        return self._data.shape[0] if self._data is not None else self.n_channels + 3

    def stop(self):
        self._data = None
        self._pos = 0


class DemoSwallowAmplifier:
    """Live synthetic 16-channel source for GUI demos.

    This class intentionally mirrors the small subset of WiFiShieldAmplifier
    used by main_window.py: start/stop, get_recent, get_srate,
    get_n_channels, is_streaming and inject_label.  It does not import MNE,
    open sockets, or write data by itself.
    """

    def __init__(self, srate: float = 500.0, n_channels: int = 16,
                 chunk_size: int = 50):
        self.srate = float(srate)
        self.n_channels = int(n_channels)
        self.chunk_size = int(chunk_size)
        self._running = False
        self._start_time = 0.0
        self._last_returned = 0
        self._sample_count = 0
        self._events: list[tuple[int, int]] = []
        self._rng = np.random.default_rng(20260703)

    def start(self):
        self._running = True
        self._start_time = time.perf_counter()
        self._last_returned = 0
        self._sample_count = 0
        self._events = []

    def stop(self):
        self._running = False

    def is_streaming(self) -> bool:
        return self._running

    def get_srate(self) -> float:
        return self.srate

    def get_n_channels(self) -> int:
        return self.n_channels

    def get_sample_count(self) -> int:
        return self._sample_count

    def inject_label(self, code: int):
        self._events.append((self._sample_count, int(code)))

    def get_recent(self, n_samples: int) -> Optional[np.ndarray]:
        if not self._running:
            return None
        expected = int((time.perf_counter() - self._start_time) * self.srate)
        n_new = expected - self._last_returned
        if n_new <= 0:
            return None
        n = min(int(n_samples), n_new)
        start = expected - n
        indexes = np.arange(start, expected, dtype=np.float64)
        self._last_returned = expected
        self._sample_count = expected
        return self._generate(indexes)

    def _generate(self, indexes: np.ndarray) -> np.ndarray:
        t = indexes / self.srate
        n = t.size
        data = self._rng.normal(0.0, 0.9, size=(self.n_channels, n))

        # EEG channels: FP1, FP2, C3, C4, P7, P8, F3, F4, Cz.
        eeg_channels = [0, 1, 2, 3, 4, 5, 10, 11, 13]
        for offset, ch in enumerate(eeg_channels):
            if ch >= self.n_channels:
                continue
            alpha = 12.0 * np.sin(2 * np.pi * 10.0 * t + offset * 0.45)
            beta = 4.5 * np.sin(2 * np.pi * 18.0 * t + offset * 0.23)
            drift = 2.0 * np.sin(2 * np.pi * 0.25 * t + offset)
            data[ch] += alpha + beta + drift

        # EMG channels: throat and chest electrodes.  More high-frequency
        # content than EEG, with bursts near paradigm labels.
        emg_channels = [6, 7, 8, 9, 14, 15]
        burst = self._event_envelope(indexes)
        for offset, ch in enumerate(emg_channels):
            if ch >= self.n_channels:
                continue
            carrier = np.sin(2 * np.pi * (62 + offset * 7) * t)
            baseline = 4.0 * np.sin(2 * np.pi * 28.0 * t + offset)
            data[ch] += baseline + 35.0 * burst * carrier

        # ECG-like channel at CH13.  It is visible but intentionally modest,
        # because many demos may not have real ECG electrodes connected.
        if self.n_channels > 12:
            heart_phase = (t * 1.15) % 1.0
            qrs = 70.0 * np.exp(-((heart_phase - 0.04) ** 2) / (2 * 0.004))
            p_wave = 8.0 * np.exp(-((heart_phase - 0.78) ** 2) / (2 * 0.002))
            data[12] += qrs + p_wave + 3.0 * np.sin(2 * np.pi * 1.15 * t)

        return data.astype(np.float32)

    def _event_envelope(self, indexes: np.ndarray) -> np.ndarray:
        if not self._events:
            return np.zeros(indexes.shape, dtype=np.float64)
        env = np.zeros(indexes.shape, dtype=np.float64)
        for sample, code in self._events[-20:]:
            if code not in (2, 3, 4):
                continue
            center = sample + int(1.2 * self.srate)
            sigma = 0.35 * self.srate
            env += np.exp(-((indexes - center) ** 2) / (2 * sigma ** 2))
        return np.clip(env, 0.0, 1.8)


class OpenBCISource:
    """LSL-based OpenBCI data source.

    Wraps LSLInlet for receiving OpenBCI data via Lab Streaming Layer.

    Parameters
    ----------
    stream_name : str
        LSL stream name to resolve.
    n_channels : int
        Number of channels to read.
    srate : float
        Sampling rate in Hz.
    """

    def __init__(self, stream_name: str = "OpenBCI_GUI",
                 n_channels: int = 8, srate: float = 250.0):
        self.stream_name = stream_name
        self.n_channels = n_channels
        self.srate = srate
        self._inlet = None
        self._connected = False
        self._buffer = []
        self._markers = []

    def start(self):
        """Connect to LSL stream."""
        try:
            import pylsl
            streams = pylsl.resolve_byprop("name", self.stream_name, timeout=5)
            if not streams:
                logger.warning("LSL stream '%s' not found", self.stream_name)
                return
            self._inlet = pylsl.StreamInlet(streams[0])
            self._connected = True
            logger.info("Connected to LSL stream: %s", self.stream_name)
        except Exception as e:
            logger.error("LSL connection failed: %s", e)

    def is_connected(self) -> bool:
        return self._connected

    def get_recent(self, n_samples: int) -> Optional[np.ndarray]:
        """Read recent samples from LSL stream."""
        if not self._connected or self._inlet is None:
            return None
        try:
            samples = []
            for _ in range(n_samples):
                sample, timestamp = self._inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break
                samples.append(sample[:self.n_channels])
            if not samples:
                return None
            return np.array(samples, dtype=np.float32).T  # (n_ch, n)
        except Exception:
            return None

    def pop_markers(self) -> list:
        """Get and clear accumulated markers."""
        markers = list(self._markers)
        self._markers.clear()
        return markers

    def inject_label(self, code: int):
        """Inject an event label."""
        self._markers.append((time.time(), code))

    def get_srate(self) -> float:
        return self.srate

    def get_n_channels(self) -> int:
        return self.n_channels

    def is_streaming(self) -> bool:
        return self._connected

    def stop(self):
        self._connected = False
        self._inlet = None


class WiFiShieldAmplifier:
    """OpenBCI WiFi Shield source using HTTP control plus TCP data relay.

    The Shield does not stream EEG on port 80. Port 80 is only used for HTTP
    control; after ``/stream/start`` the Shield connects back to this computer
    on ``port`` and sends raw OpenBCI packets.
    """

    PACKET_SIZE = 33
    VREF = 4.5
    ADS_SCALE = float((2 ** 23) - 1)

    def __init__(self, host: str = "192.168.4.1", port: int = 9000,
                 n_channels: int = 16, srate: float = 500.0,
                 gain: int = 24, buffer_seconds: float = 60.0,
                 auto_channels: bool = False,
                 configure_channels: bool = True,
                 channel_gain: int = 6):
        self.host = host
        self.port = int(port)
        self.requested_n_channels = int(n_channels)
        self.n_channels = int(n_channels)
        self.auto_channels = bool(auto_channels)
        self.configure_channels = bool(configure_channels)
        self.channel_gain = int(channel_gain)
        self.srate = float(srate)
        self.gain = int(gain)
        self._server_socket: Optional[socket.socket] = None
        self._data_socket: Optional[socket.socket] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._buffer = deque(maxlen=max(1, int(self.srate * buffer_seconds)))
        self._rx = bytearray()
        self._lock = threading.Lock()
        self._sample_count = 0
        self._last_returned = 0
        self._pending_8ch: Optional[np.ndarray] = None
        self._gains = [self.gain] * 8

    def start(self):
        """Configure the Shield and wait for its TCP data connection."""
        self.stop()
        local_ip = self._get_local_ip()
        bind_ip = local_ip
        logger.info(
            "WiFiShield: %s:80 -> TCP:%s, %sch, x%s, %sHz",
            self.host, self.port, self.n_channels, self.gain, int(self.srate),
        )
        logger.info("local=%s:%s, device=%s:80", local_ip, self.port, self.host)

        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self._server_socket.bind((bind_ip, self.port))
            except OSError:
                self._server_socket.bind(("", self.port))
            self._server_socket.listen(1)
            self._server_socket.settimeout(10.0)

            self._read_board_info()
            self._configure_tcp(local_ip)
            self._http_request("GET", "/stream/stop", timeout=1.5, ignore_errors=True)
            if self.configure_channels:
                self._configure_cyton_channels()
            self._http_request("GET", "/stream/start", timeout=3.0)
            logger.info("Sent /stream/start, waiting for Shield TCP connection...")

            self._data_socket, addr = self._server_socket.accept()
            self._data_socket.settimeout(0.5)
            self._running = True
            logger.info("Shield connected: %s:%s", addr[0], addr[1])

            self._reader_thread = threading.Thread(
                target=self._read_loop, name="WiFiShieldReader", daemon=True)
            self._reader_thread.start()
        except Exception as exc:
            logger.error("WiFi Shield connection failed: %s", exc)
            self.stop()
            raise

    def _get_local_ip(self) -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect((self.host, 80))
            return sock.getsockname()[0]
        except Exception:
            return "0.0.0.0"
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _http_request(self, method: str, path: str, payload: Optional[dict] = None,
                      timeout: float = 3.0, ignore_errors: bool = False):
        url = f"http://{self.host}{path}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if ignore_errors:
                return None
            raise exc
        if not raw:
            return None
        text = raw.decode("utf-8", errors="replace")
        try:
            return json.loads(text)
        except ValueError:
            return text

    def _read_board_info(self):
        info = self._http_request("GET", "/board", timeout=3.0, ignore_errors=True)
        if isinstance(info, dict):
            board_channels = info.get("num_channels")
            try:
                board_channels = int(board_channels)
            except (TypeError, ValueError):
                board_channels = 0
            if self.auto_channels and board_channels > 0 and board_channels < self.n_channels:
                logger.warning(
                    "Shield reports %d channels; using that instead of requested %d to avoid fake channels",
                    board_channels, self.n_channels,
                )
                self.n_channels = board_channels
                self._pending_8ch = None
            gains = info.get("gains")
            if isinstance(gains, list) and gains:
                self._gains = [int(g) if int(g) > 0 else self.gain for g in gains[:8]]
            logger.info("Shield info: %s", info)

    def _configure_tcp(self, local_ip: str):
        payloads = [
            {"ip": local_ip, "port": self.port, "output": "raw"},
            {"ip": local_ip, "port": self.port, "delimiter": True, "latency": 1000, "output": "raw"},
        ]
        last_error = None
        for payload in payloads:
            try:
                self._http_request("POST", "/tcp", payload=payload, timeout=3.0)
                return
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            logger.warning("Could not update Shield TCP target: %s", last_error)

    def _send_board_command(self, command: str) -> bool:
        command = str(command)
        attempts = [
            ("POST", "/command", {"command": command}),
            ("GET", "/command?command=" + urllib.parse.quote(command), None),
        ]
        for method, path, payload in attempts:
            try:
                self._http_request(method, path, payload=payload, timeout=2.0)
                return True
            except Exception:
                continue
        logger.warning("Could not send Cyton command: %s", command)
        return False

    def _configure_cyton_channels(self):
        """Put Cyton channels back to normal input and x6 gain before streaming."""
        gain_codes = {1: "0", 2: "1", 4: "2", 6: "3", 8: "4", 12: "5", 24: "6"}
        gain_code = gain_codes.get(int(self.channel_gain), "3")
        self._send_board_command("d")
        time.sleep(0.05)
        channel_ids = "12345678QWERTYUI"[:max(1, min(self.n_channels, 16))]
        for ch in channel_ids:
            # x <channel> <power=on> <gain> <normal input> <bias> <srb2> <srb1> X
            self._send_board_command(f"x{ch}0{gain_code}0110X")
            time.sleep(0.02)
        logger.info("Cyton channels configured: normal input, x%s", self.channel_gain)
    def _read_loop(self):
        while self._running and self._data_socket is not None:
            try:
                chunk = self._data_socket.recv(4096)
                if not chunk:
                    break
                self._rx.extend(chunk)
                self._parse_rx()
            except socket.timeout:
                continue
            except OSError:
                break
            except Exception as exc:
                logger.debug("WiFi reader error: %s", exc)
                break
        self._running = False

    def _parse_rx(self):
        while self._rx:
            start = self._rx.find(b"\xa0")
            if start >= 0:
                if start:
                    del self._rx[:start]
                if len(self._rx) < self.PACKET_SIZE:
                    return
                packet = bytes(self._rx[:self.PACKET_SIZE])
                if (packet[-1] & 0xF0) != 0xC0:
                    del self._rx[0]
                    continue
                del self._rx[:self.PACKET_SIZE]
                self._append_openbci_packet(packet)
                continue

            sample_bytes = self.n_channels * 4
            if len(self._rx) < sample_bytes:
                return
            n_floats = (len(self._rx) // 4 // self.n_channels) * self.n_channels
            if n_floats <= 0:
                return
            raw = bytes(self._rx[:n_floats * 4])
            del self._rx[:n_floats * 4]
            floats = np.frombuffer(raw, dtype=np.float32).reshape(-1, self.n_channels)
            for sample in floats:
                self._append_sample(sample)

    def _append_openbci_packet(self, packet: bytes):
        chans = []
        for idx in range(8):
            offset = 2 + idx * 3
            value = int.from_bytes(packet[offset:offset + 3], byteorder="big", signed=False)
            if value & 0x800000:
                value -= 0x1000000
            gain = self._gains[idx] if idx < len(self._gains) else self.gain
            gain = gain if gain > 0 else self.gain
            chans.append(value * (self.VREF / gain / self.ADS_SCALE) * 1e6)
        chans = np.asarray(chans, dtype=np.float32)

        if self.n_channels <= 8:
            self._append_sample(chans[:self.n_channels])
            return
        if self._pending_8ch is None:
            self._pending_8ch = chans
            return
        sample = np.concatenate([self._pending_8ch, chans])[:self.n_channels]
        self._pending_8ch = None
        self._append_sample(sample)

    def _append_sample(self, sample: np.ndarray):
        sample = np.asarray(sample, dtype=np.float32).reshape(-1)
        if sample.size < self.n_channels:
            sample = np.pad(sample, (0, self.n_channels - sample.size), constant_values=np.nan)
        elif sample.size > self.n_channels:
            sample = sample[:self.n_channels]
        with self._lock:
            self._buffer.append(sample)
            self._sample_count += 1

    def get_recent(self, n_samples: int) -> Optional[np.ndarray]:
        """Return newly received samples as (n_channels, n_samples)."""
        if not self._running:
            return None
        with self._lock:
            n_new = self._sample_count - self._last_returned
            if n_new <= 0 or len(self._buffer) == 0:
                return None
            n = min(int(n_samples), n_new, len(self._buffer))
            rows = list(self._buffer)[-n:]
            self._last_returned = self._sample_count
        return np.asarray(rows, dtype=np.float32).T

    def get_sample_count(self) -> int:
        with self._lock:
            return self._sample_count

    def get_srate(self) -> float:
        return self.srate

    def get_n_channels(self) -> int:
        return self.n_channels

    def is_streaming(self) -> bool:
        return self._running

    def inject_label(self, code: int):
        pass

    def stop(self):
        self._running = False
        self._http_request("GET", "/stream/stop", timeout=1.0, ignore_errors=True)
        for attr in ("_data_socket", "_server_socket"):
            sock = getattr(self, attr)
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
                setattr(self, attr, None)
        if (self._reader_thread is not None
                and self._reader_thread is not threading.current_thread()):
            self._reader_thread.join(timeout=1.0)
        self._reader_thread = None

def create_source(source_type: str = "sim", **kwargs):
    """Factory function for creating data sources.

    Parameters
    ----------
    source_type : str
        One of "sim", "lsl", "wifi".
    **kwargs
        Passed to the source constructor.
    """
    if source_type == "sim":
        return SimSwallowAmplifier(**kwargs)
    elif source_type == "demo":
        return DemoSwallowAmplifier(**kwargs)
    elif source_type == "lsl":
        return OpenBCISource(**kwargs)
    elif source_type == "wifi":
        return WiFiShieldAmplifier(**kwargs)
    else:
        raise ValueError(f"Unknown source type: {source_type}")
