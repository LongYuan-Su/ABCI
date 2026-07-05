# -*- coding: utf-8 -*-
"""MetaBCI brainflow: acquisition, processing, control, and GUI helpers."""

from importlib import import_module

from .amplifiers import (
    BaseAmplifier,
    HTOnlineSystem,
    LSLapps,
    Marker,
    RingBuffer,
)
from .logger import disable_log, get_logger
from .workers import ContinuousWorker, ProcessWorker

_LAZY_EXPORTS = {
    "DBManager": ("metabci.brainflow.acquisition.database", "DBManager"),
    "DemoSwallowAmplifier": ("metabci.brainflow.acquisition.sources", "DemoSwallowAmplifier"),
    "EEGRecorder": ("metabci.brainflow.acquisition.recorder", "EEGRecorder"),
    "OpenBCISource": ("metabci.brainflow.acquisition.sources", "OpenBCISource"),
    "PatientRecord": ("metabci.brainflow.acquisition.database", "PatientRecord"),
    "RealTimeBuffer": ("metabci.brainflow.acquisition.sources", "RealTimeBuffer"),
    "SimSwallowAmplifier": ("metabci.brainflow.acquisition.sources", "SimSwallowAmplifier"),
    "SwallowDataSimulator": ("metabci.brainflow.acquisition.simulator", "SwallowDataSimulator"),
    "WiFiShieldAmplifier": ("metabci.brainflow.acquisition.sources", "WiFiShieldAmplifier"),
    "create_source": ("metabci.brainflow.acquisition.sources", "create_source"),
    "LSLStimulator": ("metabci.brainflow.control.stimulator", "LSLStimulator"),
    "OnlineSwallowIntentDetector": (
        "metabci.brainflow.control.online_swallow_control",
        "OnlineSwallowIntentDetector",
    ),
    "PrintStimulator": ("metabci.brainflow.control.stimulator", "PrintStimulator"),
    "SerialStimulator": ("metabci.brainflow.control.stimulator", "SerialStimulator"),
    "SwallowClosedLoop": ("metabci.brainflow.control.closed_loop", "SwallowClosedLoop"),
    "create_stimulator": ("metabci.brainflow.control.stimulator", "create_stimulator"),
    "FeatureExtractor": ("metabci.brainflow.processing.feature_extraction", "FeatureExtractor"),
    "SignalQualityMonitor": ("metabci.brainflow.processing.signal_quality", "SignalQualityMonitor"),
    "SwallowAssessmentEngine": ("metabci.brainflow.processing.assessment", "SwallowAssessmentEngine"),
    "assess_from_paradigm_log": (
        "metabci.brainflow.processing.assessment",
        "assess_from_paradigm_log",
    ),
    "create_decoder": ("metabci.brainflow.processing.decoder", "create_decoder"),
}

__all__ = [
    "BaseAmplifier",
    "HTOnlineSystem",
    "LSLapps",
    "Marker",
    "RingBuffer",
    "disable_log",
    "get_logger",
    "ContinuousWorker",
    "ProcessWorker",
    *_LAZY_EXPORTS.keys(),
]


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
