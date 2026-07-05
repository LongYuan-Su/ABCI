# -*- coding: utf-8 -*-
"""Swallow BCI dataset — OpenBCI BDF recordings with imagined and water swallow.

TODO (Fork C integration): Implement BDF loading and trial segmentation
from 本科生_竞赛1/样本*_特征提取.py, using ``metabci.brainda.datasets.base.BaseDataset``.

Channel layout (16-channel OpenBCI, 500 Hz):
  - EEG: [0, 1, 2, 3, 4, 5, 10, 11, 13]  (9 channels)
  - EMG: [6, 7, 8, 9, 14, 15]             (6 channels)
  - ECG: [12]                                (1 channel)
"""

from __future__ import annotations

from ..base import BaseDataset


class SwallowDataset(BaseDataset):
    """Swallow motor imagery + water swallow dataset from OpenBCI BDF.

    Events
    ------
    imagine_swallow : (1, (0.0, 5.0))
        5-second imagined swallow trials (0-3s label=swallow, 3-5s label=rest).
    water_swallow : (2, (0.0, 15.0))
        15-second warm-water swallow trials (0-10s label=swallow, 10-15s label=rest).
    """

    _CHANNELS_EEG = [0, 1, 2, 3, 4, 5, 10, 11, 13]
    _CHANNELS_EMG = [6, 7, 8, 9, 14, 15]
    _CHANNELS_ECG = [12]
    _SRATE = 500.0

    def __init__(self):
        super().__init__(
            dataset_code="swallow_bci",
            subjects=[1, 2, 3, 4],
            events={
                "imagine_swallow": (1, (0.0, 5.0)),
                "water_swallow": (2, (0.0, 15.0)),
            },
            channels=[
                f"Ch{i + 1}"
                for i in range(16)
            ],
            srate=self._SRATE,
            paradigm="swallow_imagery",
        )

    def _get_single_subject_data(self, subject_idx: int):
        """TODO: Port from 本科生_竞赛1 BDF pipeline.

        Steps:
          1. Read BDF file via ``mne.io.read_raw_bdf``
          2. Apply 4-30 Hz bandpass filter
          3. Resample to 500 Hz
          4. Segment into imagined + water swallow trials per timestamps
          5. Return ``mne.io.RawArray`` or ``mne.Epochs``
        """
        raise NotImplementedError(
            "SwallowDataset data loading is not yet ported from Fork C."
        )
