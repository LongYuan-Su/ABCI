# -*- coding: utf-8 -*-
"""Closed-loop and stimulation control helpers."""

from importlib import import_module

_LAZY_EXPORTS = {
    "SwallowClosedLoop": ("metabci.brainflow.control.closed_loop", "SwallowClosedLoop"),
    "OnlineSwallowIntentDetector": (
        "metabci.brainflow.control.online_swallow_control",
        "OnlineSwallowIntentDetector",
    ),
    "LSLStimulator": ("metabci.brainflow.control.stimulator", "LSLStimulator"),
    "PrintStimulator": ("metabci.brainflow.control.stimulator", "PrintStimulator"),
    "SerialStimulator": ("metabci.brainflow.control.stimulator", "SerialStimulator"),
    "create_stimulator": ("metabci.brainflow.control.stimulator", "create_stimulator"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
