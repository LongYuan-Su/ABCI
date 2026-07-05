# -*- coding: utf-8 -*-
"""Swallow imagery paradigm.

TODO (Fork C integration): Port trial segmentation logic from
本科生_竞赛1, using ``metabci.brainda.paradigms.base.BaseParadigm``.

Trial structure (per epoch):
  - imagined swallow: 0-5 s window, 0-3 s label=1 (swallow), 3-5 s label=0 (rest)
  - water swallow:  0-15 s window, 0-10 s label=1, 10-15 s label=0
"""

from __future__ import annotations

from ..base import BaseParadigm


class SwallowImagery(BaseParadigm):
    """Swallow motor imagery paradigm with imagined and water-swallow trials.

    Compatible with ``SwallowDataset`` (dataset_code="swallow_bci").
    """

    def is_valid(self, dataset) -> bool:
        return dataset.paradigm == "swallow_imagery"
