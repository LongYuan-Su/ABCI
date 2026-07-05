from .paradigm import SSVEP, P300, AVEP, MI
from .framework import Experiment

# Swallow paradigm — standalone + Experiment-based
from .paradigm_swallow import (
    run_paradigm as run_swallow_paradigm,
    run_swallow_paradigm_via_experiment,
)


from .paradigm_swallow_control import (
    run_control_paradigm as run_swallow_control_paradigm,
)

__all__ = [
    "SSVEP", "P300", "AVEP", "MI",
    "Experiment",
    "run_swallow_paradigm",
    "run_swallow_paradigm_via_experiment",
    "run_swallow_control_paradigm",
]
