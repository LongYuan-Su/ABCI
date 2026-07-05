# -*- coding: utf-8 -*-
"""Decoder factory for MetaBCI brainflow — powered by brainda algorithms.

All decoders delegate to ``metabci.brainda.algorithms`` implementations:
  - ``eegnet``       → ``brainda.algorithms.deep_learning.eegnet.EEGNet`` (Lawhern 2018)
  - ``riemann_mdm``   → ``brainda.algorithms.manifold.riemann.MDRM`` (tangent-space LDA)
  - ``swallow_classifier`` → Fork C placeholder (ReplacedThreeBranchSwallowNet)

Examples
--------
>>> from metabci.brainflow.processing.decoder import create_decoder
>>> dec = create_decoder("eegnet", n_channels=8, n_samples=200, n_classes=2)
>>> dec.fit(X_train, y_train)
>>> preds = dec.predict(X_test)
"""

try:
    from ..logger import get_logger
except ImportError:
    from metabci.brainflow.logger import get_logger  # type: ignore[no-redef]

logger = get_logger("decoder")


def create_decoder(decoder_name: str = "eegnet", **kwargs):
    """Factory function for creating decoder instances.

    All decoders are sourced from ``metabci.brainda.algorithms``.

    Parameters
    ----------
    decoder_name : str
        One of ``"eegnet"``, ``"riemann_mdm"``, ``"swallow_classifier"``.
    **kwargs
        Forwarded to the underlying brainda constructor.

    Returns
    -------
    decoder
        Object with ``fit()``, ``predict()``, ``predict_proba()`` interface.
    """
    decoder_name = decoder_name.lower()

    if decoder_name == "eegnet":
        return _create_eegnet(**kwargs)
    elif decoder_name == "riemann_mdm":
        return _create_riemann_mdm(**kwargs)
    elif decoder_name == "swallow_classifier":
        return _create_swallow_classifier(**kwargs)
    else:
        raise ValueError(
            f"Unknown decoder '{decoder_name}'. "
            f"Available: eegnet, riemann_mdm, swallow_classifier"
        )


# ---------------------------------------------------------------------------
# EEGNet — delegate to brainda.algorithms.deep_learning.eegnet.EEGNet
# ---------------------------------------------------------------------------

def _create_eegnet(**kwargs):
    """Create EEGNet via brainda's canonical Lawhern implementation.

    ``metabci.brainda.algorithms.deep_learning.eegnet.EEGNet`` is a
    ``@SkorchNet``-wrapped ``NeuralNetClassifier`` with proper separable
    convolutions, MaxNorm constraint, batch normalisation, and dropout.
    """
    try:
        from metabci.brainda.algorithms.deep_learning.eegnet import EEGNet
    except ImportError:
        raise ImportError(
            "brainda EEGNet is not available. "
            "Install metabci with brainda extras: pip install metabci[brainda]"
        )

    n_channels = kwargs.pop("n_channels", 8)
    n_samples = kwargs.pop("n_samples", 200)
    n_classes = kwargs.pop("n_classes", 2)

    logger.info(
        "Creating brainda EEGNet: n_channels=%d, n_samples=%d, n_classes=%d",
        n_channels, n_samples, n_classes,
    )
    return EEGNet(
        n_channels=n_channels,
        n_classes=n_classes,
        input_window_samples=n_samples,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Riemannian MDM — delegate to brainda.algorithms.manifold.riemann.MDRM
# ---------------------------------------------------------------------------

def _create_riemann_mdm(**kwargs):
    """Create Riemannian MDRM via brainda's tangent-space LDA classifier.

    ``metabci.brainda.algorithms.manifold.riemann.MDRM`` is a full
    sklearn-compatible ``BaseEstimator/TransformerMixin/ClassifierMixin``
    that projects covariance matrices to tangent space then applies LDA.
    """
    try:
        from metabci.brainda.algorithms.manifold.riemann import MDRM
    except ImportError:
        raise ImportError(
            "brainda MDRM is not available. "
            "Install metabci with brainda extras: pip install metabci[brainda]"
        )

    logger.info("Creating brainda MDRM Riemannian classifier")
    return MDRM(**kwargs)


# ---------------------------------------------------------------------------
# Swallow Classifier — Fork C placeholder
# ---------------------------------------------------------------------------

def _create_swallow_classifier(**kwargs):
    """Create SwallowClassifier via brainda's ReplacedThreeBranchSwallowNet.

    ``metabci.brainda.algorithms.deep_learning.swallow_net.ReplacedThreeBranchSwallowNet``
    is the 3-modality × 3-branch classifier ported from Fork C.
    """
    from metabci.brainda.algorithms.deep_learning.swallow_net import (
        ReplacedThreeBranchSwallowNet)

    checkpoint = kwargs.pop("checkpoint", None)
    if checkpoint:
        from metabci.brainda.algorithms.deep_learning.swallow_trainer import (
            load_swallow_classifier)
        return load_swallow_classifier(checkpoint)

    return ReplacedThreeBranchSwallowNet(**kwargs)
