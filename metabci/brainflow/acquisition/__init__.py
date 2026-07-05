# -*- coding: utf-8 -*-
"""Data sources, recording, runtime database, and data viewing helpers."""

from importlib import import_module

_LAZY_EXPORTS = {
    "DBManager": ("metabci.brainflow.acquisition.database", "DBManager"),
    "PatientRecord": ("metabci.brainflow.acquisition.database", "PatientRecord"),
    "EEGRecorder": ("metabci.brainflow.acquisition.recorder", "EEGRecorder"),
    "SwallowDataSimulator": ("metabci.brainflow.acquisition.simulator", "SwallowDataSimulator"),
    "DemoSwallowAmplifier": ("metabci.brainflow.acquisition.sources", "DemoSwallowAmplifier"),
    "OpenBCISource": ("metabci.brainflow.acquisition.sources", "OpenBCISource"),
    "RealTimeBuffer": ("metabci.brainflow.acquisition.sources", "RealTimeBuffer"),
    "SimSwallowAmplifier": ("metabci.brainflow.acquisition.sources", "SimSwallowAmplifier"),
    "WiFiShieldAmplifier": ("metabci.brainflow.acquisition.sources", "WiFiShieldAmplifier"),
    "create_source": ("metabci.brainflow.acquisition.sources", "create_source"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
