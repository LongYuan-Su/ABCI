"""MetaBCI brainflow GUI — 吞咽调控实验控制中心."""

from .main_window import MainWindow, main
from .patient_dialogs import PatientEditDialog, HistoryDialog
from .score_dialog import (
    DecoderResultDialog,
    ClosedLoopResultDialog,
    AssessmentReportDialog,
)
from .eeg_display import (
    MultiRegionEEGWidget,
    LabelPanel,
    DEFAULT_LABELS,
    HAS_PYQTGRAPH,
)
