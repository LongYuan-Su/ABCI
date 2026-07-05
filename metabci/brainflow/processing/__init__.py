# -*- coding: utf-8 -*-
"""Signal processing, assessment, decoding, and signal-quality helpers."""

from importlib import import_module

_LAZY_EXPORTS = {
    "SwallowAssessmentEngine": ("metabci.brainflow.processing.assessment", "SwallowAssessmentEngine"),
    "assess_from_paradigm_log": ("metabci.brainflow.processing.assessment", "assess_from_paradigm_log"),
    "create_decoder": ("metabci.brainflow.processing.decoder", "create_decoder"),
    "FeatureExtractor": ("metabci.brainflow.processing.feature_extraction", "FeatureExtractor"),
    "SignalQualityMonitor": ("metabci.brainflow.processing.signal_quality", "SignalQualityMonitor"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
