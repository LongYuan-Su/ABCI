# -*- coding: utf-8 -*-
"""Training / inference wrappers for swallow BCI models.

Provides sklearn-compatible wrappers and checkpoint loaders for the
models in ``metabci.brainda.algorithms.deep_learning.swallow_net``.

Examples
--------
>>> from metabci.brainda.algorithms.deep_learning.swallow_trainer import (
...     load_swallow_classifier, load_swallow_quantifier)
>>> model = load_swallow_classifier("model.pt")
>>> prob = torch.sigmoid(model(eeg, emg, ecg))
"""

from __future__ import annotations

import torch


def load_swallow_classifier(path: str, device: str = "cpu"):
    """Load a trained ReplacedThreeBranchSwallowNet from a .pt checkpoint.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint file.
    device : str
        Torch device string.

    Returns
    -------
    model : ReplacedThreeBranchSwallowNet
        Loaded model in eval mode.
    """
    from metabci.brainda.algorithms.deep_learning.swallow_net import (
        ReplacedThreeBranchSwallowNet)

    pkg = torch.load(path, map_location=device)
    cfg_dict = pkg.get("cfg", {})
    model = ReplacedThreeBranchSwallowNet(**cfg_dict)
    model.load_state_dict(pkg["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_swallow_quantifier(path: str, device: str = "cpu"):
    """Load a trained SwallowQuantificationNet from a .pt checkpoint.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint file.
    device : str
        Torch device string.

    Returns
    -------
    model : SwallowQuantificationNet
        Loaded model in eval mode.
    """
    from metabci.brainda.algorithms.deep_learning.swallow_net import (
        SwallowQuantificationNet)

    pkg = torch.load(path, map_location=device)
    cfg_dict = pkg.get("cfg", {})
    model = SwallowQuantificationNet(**cfg_dict)
    model.load_state_dict(pkg["model_state_dict"])
    model.to(device)
    model.eval()
    return model
