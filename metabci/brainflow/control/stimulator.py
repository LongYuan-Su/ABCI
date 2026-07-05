# -*- coding: utf-8 -*-
"""Stimulator classes and factory for MetaBCI brainflow.

Port interface aligns with ``metabci.brainstim.utils`` (NeuroScanPort /
NeuraclePort / LsLPort) ``setData(label)`` convention.  LSL outlet
creation follows the same ``StreamInfo`` / ``StreamOutlet`` pattern as
``metabci.brainflow.workers.ProcessWorker``.

Provides: PrintStimulator, SerialStimulator, LSLStimulator, create_stimulator

References
----------
- ``metabci.brainstim.utils.NeuroScanPort.setData``  — serial/parallel trigger
- ``metabci.brainstim.utils.NeuraclePort.setData``     — Neuracle trigger box
- ``metabci.brainflow.workers.ProcessWorker.pre()``     — LSL outlet pattern
"""

import time
import threading
from abc import ABC, abstractmethod

from ..logger import get_logger

logger = get_logger("stimulator")


class BaseStimulator(ABC):
    """Abstract base class for stimulators."""

    @abstractmethod
    def trigger(self, duration: float = 0.5, intensity: float = 1.0, **kwargs):
        """Trigger stimulation."""

    @abstractmethod
    def stop(self):
        """Stop stimulation."""


class PrintStimulator(BaseStimulator):
    """Print-based stimulator (simulation)."""

    def trigger(self, duration: float = 0.5, intensity: float = 1.0, **kwargs):
        msg = (
            f"[电刺激] 触发! 强度={intensity:.2f}, 持续={duration:.1f}s"
        )
        print(msg)
        logger.info(msg)

    def stop(self):
        print("[电刺激] 停止")


class SerialStimulator(BaseStimulator):
    """Serial-port stimulator (placeholder)."""

    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._serial = None
        self._connect()

    def _connect(self):
        try:
            import serial
            self._serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)
            logger.info("Serial stimulator connected: %s", self.port)
        except Exception as e:
            logger.warning("Serial stimulator connection failed: %s", e)
            self._serial = None

    def trigger(self, duration: float = 0.5, intensity: float = 1.0, **kwargs):
        if self._serial is None:
            logger.warning("Serial not connected, skipping trigger")
            return
        try:
            val = max(0, min(255, int(intensity * 255)))
            self._serial.write(bytes([val]))
            threading.Timer(duration, self._serial.write, [bytes([0])]).start()
        except Exception as e:
            logger.error("Serial trigger failed: %s", e)

    def stop(self):
        if self._serial:
            try:
                self._serial.write(bytes([0]))
                self._serial.close()
            except Exception:
                pass
            self._serial = None


class LSLStimulator(BaseStimulator):
    """LSL outlet stimulator."""

    def __init__(self, stream_name: str = "StimulusMarkers"):
        self.stream_name = stream_name
        self._outlet = None
        self._init_lsl()

    def _init_lsl(self):
        try:
            import pylsl
            info = pylsl.StreamInfo(
                self.stream_name, "Markers", 1,
                pylsl.IRREGULAR_RATE, pylsl.cf_int32,
                f"stimulator_{id(self)}",
            )
            self._outlet = pylsl.StreamOutlet(info)
            logger.info("LSL stimulator outlet created: %s", self.stream_name)
        except Exception as e:
            logger.warning("LSL stimulator init failed: %s", e)

    def trigger(self, duration: float = 0.5, intensity: float = 1.0, **kwargs):
        if self._outlet:
            try:
                self._outlet.push_sample([int(intensity * 100)])
            except Exception as e:
                logger.error("LSL trigger failed: %s", e)

    def stop(self):
        self._outlet = None


STIMULATOR_REGISTRY = {
    "print": PrintStimulator,
    "serial": SerialStimulator,
    "lsl": LSLStimulator,
}


def create_stimulator(stim_type: str = "print", **kwargs) -> BaseStimulator:
    """Factory function for creating stimulator instances.

    Parameters
    ----------
    stim_type : str
        One of "print", "serial", "lsl".
    **kwargs
        Passed to the stimulator constructor.

    Returns
    -------
    BaseStimulator
    """
    cls = STIMULATOR_REGISTRY.get(stim_type)
    if cls is None:
        logger.warning("Unknown stimulator type '%s', falling back to PrintStimulator", stim_type)
        cls = PrintStimulator
    return cls(**kwargs)
