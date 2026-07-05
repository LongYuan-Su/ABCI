# -*- coding: utf-8 -*-
"""Swallow data simulator — thin re-export from brainda.datasets.

The canonical ``SwallowDataSimulator`` lives at
``metabci.brainda.datasets.simulated_swallow.SwallowDataSimulator``
and inherits from ``metabci.brainda.datasets.base.BaseDataset``.

This module provides a convenience re-export so that acquisition/control
modules can use the simulator without reaching into brainda internals.
"""

try:
    from metabci.brainda.datasets.simulated_swallow import SwallowDataSimulator  # noqa: F401
except ImportError:
    SwallowDataSimulator = None  # type: ignore[assignment]
