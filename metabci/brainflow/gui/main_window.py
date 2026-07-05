"""脑卒中吞咽困难调控实验控制中心 — PySide6 GUI（5 页签）。

页签：
1. 患者管理 — 患者 CRUD，选中后可进行范式测试/调控评估
2. 范式评估 — 吞咽评估范式执行控制
3. 调控评估 — 吞咽想象范式执行（五次想象吞咽）
4. 实时脑电 — brainflow LSL 实时波形显示
5. 日志 — 所有子进程输出聚合

基于 MetaBCI 框架（brainflow + brainstim + brainda）构建。

属于 metabci.brainflow.gui 包（原 apps/ui/launcher_ui.py）。
"""

from __future__ import annotations

# ---- Suppress third-party warnings ----
import warnings
import os
import logging

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("PSYCHOPY_PARALLEL_PORT", "0")
logging.getLogger("psychopy").setLevel(logging.ERROR)

from typing import Optional

import json
import csv
import queue
import re
import shlex
import socket
import sys
import threading
import time
import argparse

import numpy as np
from datetime import datetime
from pathlib import Path

# ---- Cross-platform path constants (pathlib, no hardcoding) ----
_UI_DIR = Path(__file__).resolve().parent            # metabci/brainflow/gui/
_BRAINFLOW_DIR = _UI_DIR.parent                      # metabci/brainflow/
_METABCI_DIR = _BRAINFLOW_DIR.parent                 # metabci/
_PROJECT_ROOT = _METABCI_DIR.parent                  # project root

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RUNTIME_DIR = _PROJECT_ROOT / ".runtime"
DB_PATH = RUNTIME_DIR / "swallow_experiment.db"

DEFAULT_PYTHON = Path(sys.executable)
MARKER_UDP_HOST = "127.0.0.1"

# Subprocess script paths
PARADIGM_SCRIPT = _METABCI_DIR / "brainstim" / "paradigm_swallow.py"
PARADIGM_PART2_SCRIPT = _METABCI_DIR / "brainstim" / "paradigm_swallow_part2.py"
CONTROL_PARADIGM_SCRIPT = _METABCI_DIR / "brainstim" / "paradigm_swallow_control.py"
CLOSED_LOOP_SCRIPT = _BRAINFLOW_DIR / "control" / "closed_loop.py"
SWALLOW_APP_DIR = _PROJECT_ROOT / "applications" / "swallow_bci"
CONTROLLER_GUI_SCRIPT = SWALLOW_APP_DIR / "tools" / "esp32b_control_gui.py"


def _bootstrap_qt() -> None:
    """确保 PySide6 可用。"""
    try:
        import PySide6  # noqa: F401
    except ModuleNotFoundError:
        print("请安装 PySide6：pip install PySide6")
        raise SystemExit(1)


_bootstrap_qt()

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QBrush  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from metabci.brainflow.gui.patient_dialogs import PatientEditDialog, HistoryDialog
from metabci.brainflow.gui.score_dialog import (
    DecoderResultDialog, ClosedLoopResultDialog, AssessmentReportDialog,
)
from metabci.brainflow.gui.eeg_display import (
    MultiRegionEEGWidget, LabelPanel, DEFAULT_LABELS, HAS_PYQTGRAPH,
    EEG_CHANNELS, EMG_CHANNELS, ECG_CHANNELS, ALL_CHANNEL_NAMES,
)
from metabci.brainflow.processing.assessment import assess_from_paradigm_log
from metabci.brainflow.competition_algorithms import (
    run_part2_classification,
    run_warm_prior_quantification,
)
from metabci.brainflow.acquisition.sources import WiFiShieldAmplifier, DemoSwallowAmplifier
from metabci.brainflow.acquisition.recorder import EEGRecorder
from metabci.brainflow.control.online_swallow_control import OnlineSwallowIntentDetector
from applications.swallow_bci.tools.esp32b_control_gui import Esp32UdpController


# ============================================================
# 数据库简易封装（直接 sqlite3，避免 brainflow logger 依赖）
# ============================================================

class _DB:
    """轻量数据库访问（主窗口内使用）。"""

    def __init__(self, db_path: str):
        import sqlite3
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                name TEXT,
                gender TEXT,
                age INTEGER,
                height_cm REAL DEFAULT 0,
                weight_kg REAL DEFAULT 0,
                dysphagia_level TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                experiment_date TEXT NOT NULL,
                epoch_count INTEGER DEFAULT 20,
                paradigm_type TEXT DEFAULT 'swallow_assessment',
                notes TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                    ON UPDATE CASCADE ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS swallow_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                event_name TEXT NOT NULL,
                timestamp_sec REAL,
                marker_label TEXT,
                created_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                    ON UPDATE CASCADE ON DELETE CASCADE
            );
        """)
        # Migrate existing DB: add height_cm / weight_kg columns if missing
        for col, col_type in [("height_cm", "REAL DEFAULT 0"),
                               ("weight_kg", "REAL DEFAULT 0")]:
            try:
                self.conn.execute(
                    f"ALTER TABLE patients ADD COLUMN {col} {col_type};")
            except Exception:
                pass  # column already exists
        self.conn.commit()

    def get_patients(self): return self.conn.execute(
        "SELECT * FROM patients ORDER BY patient_id ASC;").fetchall()

    def get_patient(self, pid): return self.conn.execute(
        "SELECT * FROM patients WHERE patient_id = ?;", (pid,)).fetchone()

    def upsert_patient(self, data: dict):
        existing = self.get_patient(data["patient_id"])
        if existing:
            self.conn.execute(
                "UPDATE patients SET name=?,gender=?,age=?,"
                "height_cm=?,weight_kg=?,dysphagia_level=? "
                "WHERE patient_id=?;",
                (data.get("name"), data.get("gender"), data.get("age"),
                 data.get("height_cm", 0.0), data.get("weight_kg", 0.0),
                 data.get("dysphagia_level"), data["patient_id"]))
        else:
            self.conn.execute(
                "INSERT INTO patients (patient_id,name,gender,age,"
                "height_cm,weight_kg,dysphagia_level,created_at) "
                "VALUES (?,?,?,?,?,?,?,?);",
                (data["patient_id"], data.get("name"), data.get("gender"),
                 data.get("age"),
                 data.get("height_cm", 0.0), data.get("weight_kg", 0.0),
                 data.get("dysphagia_level"),
                 data.get("created_at", datetime.now().isoformat(timespec="seconds"))))
        self.conn.commit()

    def delete_patient(self, pid):
        self.conn.execute("DELETE FROM patients WHERE patient_id = ?;", (pid,))
        self.conn.commit()

    def get_sessions(self, pid=None):
        sql = """
            SELECT s.*, (SELECT COUNT(*) FROM swallow_events e WHERE e.session_id = s.session_id) AS event_count
            FROM sessions s
        """
        if pid:
            return self.conn.execute(sql + " WHERE s.patient_id = ? ORDER BY s.experiment_date DESC;", (pid,)).fetchall()
        return self.conn.execute(sql + " ORDER BY s.experiment_date DESC;").fetchall()

    def get_patient_stats(self):
        rows = self.conn.execute("""
            SELECT p.patient_id, COUNT(s.session_id) AS session_count,
                   MAX(s.experiment_date) AS last_date
            FROM patients p LEFT JOIN sessions s ON s.patient_id = p.patient_id
            GROUP BY p.patient_id ORDER BY p.patient_id ASC;
        """).fetchall()
        stats = {}
        for r in rows:
            stats[r["patient_id"]] = {
                "session_count": r["session_count"] or 0,
                "last_date": r["last_date"] or "-",
            }
        return stats

    def close(self): self.conn.close()


# ============================================================
# 启动参数弹窗
# ============================================================

class StartParadigmDialog(QDialog):
    """范式测试参数确认弹窗。"""

    def __init__(
        self,
        patient_id: str,
        parent=None,
        title: str = "开始范式测试",
        epoch_default: int = 20,
        lsl_default: str = "Swallow_Markers",
        controller_settings: bool = False,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._result: Optional[dict] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"患者：{patient_id}，请确认范式参数："))

        form = QFormLayout()
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(1, 999)
        self.epoch_spin.setValue(epoch_default)

        self.debug_check = QCheckBox("调试模式（窗口化 + 缩短时长，不勾选则为全屏）")
        self.lsl_check = QCheckBox("启用 LSL 事件标记")
        self.lsl_check.setChecked(True)
        self.lsl_name_edit = QLineEdit(lsl_default)

        self.eeg_check = QCheckBox("启动范式时同时启动脑电采集")
        self.eeg_cmd_edit = QLineEdit()
        self.eeg_cmd_edit.setPlaceholderText("例：LabRecorder.exe -c record.cfg")

        self.controller_settings = controller_settings
        self.controller_enable_check = QCheckBox("识别到吞咽想象后自动触发 ESP32B 电刺激")
        self.controller_host_edit = QLineEdit("192.168.4.1")
        self.controller_port_spin = QSpinBox()
        self.controller_port_spin.setRange(1, 65535)
        self.controller_port_spin.setValue(3333)
        self.controller_target_combo = QComboBox()
        self.controller_target_combo.addItems(["ALL", "CH1", "CH2", "CH3", "CH4"])
        self.controller_duration_spin = QDoubleSpinBox()
        self.controller_duration_spin.setRange(0.0, 60.0)
        self.controller_duration_spin.setSingleStep(0.5)
        self.controller_duration_spin.setValue(1.0)
        self.controller_duration_spin.setSuffix(" s")

        form.addRow("Epoch 重复次数", self.epoch_spin)
        form.addRow("", self.debug_check)
        form.addRow("", self.lsl_check)
        form.addRow("LSL 标记流名称", self.lsl_name_edit)
        form.addRow("", self.eeg_check)
        form.addRow("脑电采集命令", self.eeg_cmd_edit)
        if self.controller_settings:
            form.addRow("", self.controller_enable_check)
            form.addRow("ESP32B IP", self.controller_host_edit)
            form.addRow("UDP 端口", self.controller_port_spin)
            form.addRow("触发通道", self.controller_target_combo)
            form.addRow("刺激持续时间", self.controller_duration_spin)
        layout.addLayout(form)

        self.eeg_check.toggled.connect(self.eeg_cmd_edit.setEnabled)
        self.eeg_cmd_edit.setEnabled(False)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self):
        self._result = {
            "epoch_count": self.epoch_spin.value(),
            "debug": self.debug_check.isChecked(),
            "enable_lsl": self.lsl_check.isChecked(),
            "lsl_stream_name": self.lsl_name_edit.text().strip() or "Swallow_Markers",
            "start_eeg": self.eeg_check.isChecked(),
            "eeg_command": self.eeg_cmd_edit.text().strip(),
            "enable_controller": (
                self.controller_enable_check.isChecked()
                if self.controller_settings else False
            ),
            "controller_host": self.controller_host_edit.text().strip(),
            "controller_port": self.controller_port_spin.value(),
            "controller_target": self.controller_target_combo.currentText(),
            "controller_duration": self.controller_duration_spin.value(),
        }
        self.accept()

    def result(self) -> dict:
        if self._result is None:
            raise RuntimeError("result() 在确认前不可用。")
        return self._result


# ============================================================
# 主窗口
# ============================================================

class MainWindow(QMainWindow):
    """脑卒中吞咽困难调控实验控制中心。"""

    def __init__(self, demo_mode: bool = False, demo_run: str = ""):
        super().__init__()
        self.setWindowTitle("脑卒中吞咽困难调控系统")
        self.resize(1240, 860)
        self.demo_mode = bool(demo_mode)
        self.demo_run = demo_run

        # 数据库
        db_path = RUNTIME_DIR / "demo_swallow_experiment.db" if self.demo_mode else DB_PATH
        if self.demo_mode:
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        self.db = _DB(str(db_path))

        # 状态
        self.selected_patient_id: Optional[str] = None
        self.patient_stats: dict = {}
        self._paradigm_process: Optional[QProcess] = None
        self._closed_loop_process: Optional[QProcess] = None
        self._eeg_process: Optional[QProcess] = None
        self._controller_process: Optional[QProcess] = None
        self._marker_socket: Optional[socket.socket] = None
        self._marker_thread: Optional[threading.Thread] = None
        self._marker_stop: Optional[threading.Event] = None
        self._marker_udp_port = 0

        # EEG WiFi 实时采集相关
        self._eeg_amplifier: Optional[WiFiShieldAmplifier] = None
        self._eeg_plot_timer: Optional[QTimer] = None
        self._eeg_recorder: Optional[EEGRecorder] = None
        self._recording_subdir: Optional[str] = None
        self._active_paradigm_kind: Optional[str] = None
        self._eeg_connect_thread: Optional[threading.Thread] = None
        self._eeg_connect_token: Optional[object] = None
        self._eeg_connect_result: Optional[tuple] = None
        self._eeg_connect_timer: Optional[QTimer] = None
        self._eeg_connect_timeout_timer: Optional[QTimer] = None
        self._eeg_pending_amplifier = None
        self._eeg_pending_context: dict = {}
        self._eeg_live_buffer: Optional[np.ndarray] = None
        self._eeg_live_buffer_start_sample = 0
        self._eeg_live_last_end_sample = 0
        self._eeg_live_max_seconds = 30.0

        # 在线吞咽意图分类 + ESP32B 电刺激闭环
        self._online_control_enabled = False
        self._online_detector: Optional[OnlineSwallowIntentDetector] = None
        self._online_controller: Optional[Esp32UdpController] = None
        self._online_control_settings: dict = {}
        self._online_control_token: Optional[object] = None
        self._online_pending_windows: list[dict] = []
        self._online_inference_busy = False
        self._online_trigger_busy = False
        self._online_result_queue: queue.Queue = queue.Queue()
        self._online_control_timer: Optional[QTimer] = None

        # UI
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._build_patient_tab()
        self._build_paradigm_tab()
        self._build_closed_loop_tab()
        self._build_eeg_tab()
        self._build_log_tab()

        self.tabs.addTab(self._patient_tab, "患者管理")
        self.tabs.addTab(self._paradigm_tab, "范式评估")
        self.tabs.addTab(self._closed_loop_tab, "调控评估")
        self.tabs.addTab(self._eeg_tab, "实时脑电")
        self.tabs.addTab(self._log_tab, "日志")

        self.refresh_patients()
        self._online_control_timer = QTimer(self)
        self._online_control_timer.timeout.connect(self._poll_online_control)
        self._online_control_timer.start(200)
        if self.demo_mode:
            QTimer.singleShot(0, self._setup_demo_mode)

    # ================================================================
    # Tab 1: 患者管理
    # ================================================================

    def _build_patient_tab(self):
        self._patient_tab = QWidget()
        root = QVBoxLayout(self._patient_tab)

        # 操作栏
        actions = QHBoxLayout()
        self._btn_refresh = QPushButton("刷新列表")
        self._btn_add = QPushButton("录入患者")
        self._btn_edit = QPushButton("编辑患者")
        self._btn_delete = QPushButton("删除患者")
        self._btn_history = QPushButton("查看历史")
        self._btn_history.setEnabled(False)

        actions.addWidget(self._btn_refresh)
        actions.addWidget(self._btn_add)
        actions.addWidget(self._btn_edit)
        actions.addWidget(self._btn_delete)
        actions.addWidget(self._btn_history)
        actions.addStretch(1)
        root.addLayout(actions)

        self._btn_refresh.clicked.connect(self.refresh_patients)
        self._btn_add.clicked.connect(self._add_patient)
        self._btn_edit.clicked.connect(self._edit_patient)
        self._btn_delete.clicked.connect(self._delete_patient)
        self._btn_history.clicked.connect(self._show_history)

        # 患者表格
        self._patient_table = QTableWidget(0, 7)
        self._patient_table.setHorizontalHeaderLabels([
            "患者ID", "姓名", "性别", "年龄", "身高(cm)", "体重(kg)", "吞咽障碍等级"
        ])
        self._patient_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._patient_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._patient_table.setSelectionMode(QTableWidget.SingleSelection)
        self._patient_table.verticalHeader().setVisible(False)
        self._patient_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._patient_table.itemSelectionChanged.connect(self._on_patient_selected)
        root.addWidget(self._patient_table)

        # 患者详情
        info_grp = QGroupBox("选中患者信息")
        info_grid = QGridLayout(info_grp)
        self._lbl_id = QLabel("-"); self._lbl_name = QLabel("-")
        self._lbl_gender = QLabel("-"); self._lbl_age = QLabel("-")
        self._lbl_level = QLabel("-"); self._lbl_sessions = QLabel("-")
        self._lbl_last = QLabel("-")

        info_grid.addWidget(QLabel("ID"), 0, 0); info_grid.addWidget(self._lbl_id, 0, 1)
        info_grid.addWidget(QLabel("姓名"), 0, 2); info_grid.addWidget(self._lbl_name, 0, 3)
        info_grid.addWidget(QLabel("性别"), 1, 0); info_grid.addWidget(self._lbl_gender, 1, 1)
        info_grid.addWidget(QLabel("年龄"), 1, 2); info_grid.addWidget(self._lbl_age, 1, 3)
        info_grid.addWidget(QLabel("吞咽等级"), 2, 0); info_grid.addWidget(self._lbl_level, 2, 1)
        info_grid.addWidget(QLabel("测试次数"), 2, 2); info_grid.addWidget(self._lbl_sessions, 2, 3)
        info_grid.addWidget(QLabel("最近测试"), 3, 0); info_grid.addWidget(self._lbl_last, 3, 1, 1, 3)
        root.addWidget(info_grp)

    def _log_path_note(self, path: Path) -> str:
        try:
            rel = path.relative_to(_PROJECT_ROOT)
        except ValueError:
            rel = path
        return f"imported:{rel.as_posix()}"

    def _session_exists_for_note(self, note: str) -> bool:
        row = self.db.conn.execute(
            "SELECT 1 FROM sessions WHERE notes = ? LIMIT 1;", (note,)
        ).fetchone()
        return row is not None

    def _date_from_log_path(self, path: Path) -> str:
        match = re.search(r"(\d{8})_(\d{6})", path.name)
        if match:
            try:
                dt = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
                return dt.isoformat(sep=" ", timespec="seconds")
            except Exception:
                pass
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ", timespec="seconds")

    def _paradigm_type_from_log_path(self, path: Path) -> str:
        text = path.as_posix().lower()
        if "cupture_data_part2" in text or "part2" in text:
            return "swallow_part2"
        if "control" in text:
            return "swallow_control"
        if "paradigm" in text:
            return "swallow_assessment"
        return "imported_recording"

    def _load_events_from_labels(self, labels_path: Path) -> list[tuple[str, float | None, str]]:
        if not labels_path.is_file():
            return []
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            return []
        events = []
        for record in records if isinstance(records, list) else []:
            name = str(record.get("name", "") or "event")
            try:
                timestamp = float(record.get("timestamp_sec"))
            except Exception:
                timestamp = None
            marker = str(record.get("code", "") or "")
            events.append((name, timestamp, marker))
        return events

    def _load_events_from_csv_log(self, csv_path: Path) -> list[tuple[str, float | None, str]]:
        try:
            with open(csv_path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception:
            return []
        events = []
        for row in rows:
            name = row.get("event_name") or row.get("name") or "event"
            raw_ts = row.get("timestamp_sec") or row.get("timestamp") or ""
            try:
                timestamp = float(raw_ts)
            except Exception:
                timestamp = None
            marker = row.get("event_index") or row.get("code") or ""
            events.append((str(name), timestamp, str(marker)))
        return events

    def _epoch_count_from_events(self, events: list[tuple[str, float | None, str]]) -> int:
        epochs: set[int] = set()
        for name, _, _ in events:
            match = re.search(r"\bE(\d+)_", str(name))
            if match:
                try:
                    value = int(match.group(1))
                except Exception:
                    continue
                if value > 0:
                    epochs.add(value)
        return max(epochs) if epochs else 0

    def _load_patient_info_from_meta(self, meta_path: Path) -> dict:
        if not meta_path.is_file():
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return {}

        info = meta.get("patient_info") or meta.get("patient") or {}
        if not isinstance(info, dict):
            info = {}
        if not info and meta.get("subject_id"):
            info = {"patient_id": meta.get("subject_id")}
        return info

    def _import_patient_data(
        self,
        patient_id: str,
        patient_dir: Path,
        patient_info: dict | None = None,
        original_id: str | None = None,
        overwrite: bool = False,
    ) -> dict:
        patient_info = patient_info or {}
        existing = self.db.get_patient(patient_id)
        current = dict(existing) if existing is not None else {}

        def _incoming(key, default=None):
            value = patient_info.get(key, default)
            return default if value in ("", None) else value

        def _number(key, fallback=0.0):
            value = _incoming(key, current.get(key, fallback))
            try:
                return float(value) if value not in ("", None) else fallback
            except Exception:
                return fallback

        def _integer(key):
            value = _incoming(key, current.get(key))
            try:
                return int(value) if value not in ("", None) else None
            except Exception:
                return None

        if existing is not None and not overwrite:
            return current

        return {
            "patient_id": patient_id,
            "name": _incoming("name", current.get("name")),
            "gender": _incoming("gender", current.get("gender")),
            "age": _integer("age"),
            "height_cm": _number("height_cm", current.get("height_cm", 0.0) or 0.0),
            "weight_kg": _number("weight_kg", current.get("weight_kg", 0.0) or 0.0),
            "dysphagia_level": _incoming(
                "dysphagia_level",
                current.get("dysphagia_level")
                or (f"外部导入(原{original_id})" if original_id else "外部导入"),
            ),
            "created_at": current.get("created_at")
            or datetime.fromtimestamp(patient_dir.stat().st_mtime).isoformat(timespec="seconds"),
        }

    def _next_import_patient_id(self, original_id: str) -> str:
        patients = self.db.get_patients()
        existing_ids = {row["patient_id"] for row in patients}
        numeric_ids = []
        for pid in existing_ids:
            match = re.match(r"^(\d+)", str(pid))
            if match:
                numeric_ids.append(match.group(1))
        width = max([len(x) for x in numeric_ids] + [3])
        next_num = max([int(x) for x in numeric_ids] + [0]) + 1
        while True:
            candidate = f"{next_num:0{width}d}(原{original_id})"
            if candidate not in existing_ids:
                return candidate
            next_num += 1

    def _existing_import_alias(self, original_id: str) -> str | None:
        suffix = f"(原{original_id})"
        aliases = [
            row["patient_id"] for row in self.db.get_patients()
            if str(row["patient_id"]).endswith(suffix)
        ]
        return sorted(aliases)[0] if aliases else None

    def _resolve_import_patient_id(
        self,
        patient_id: str,
        patient_dir: Path,
        patient_info: dict,
    ) -> tuple[str, bool]:
        if self.db.get_patient(patient_id) is None:
            self.db.upsert_patient(
                self._import_patient_data(patient_id, patient_dir, patient_info)
            )
            return patient_id, True

        existing_alias = self._existing_import_alias(patient_id)
        if existing_alias:
            return existing_alias, False

        alias_id = self._next_import_patient_id(patient_id)
        choice = QMessageBox.question(
            self,
            "患者编号冲突",
            (
                f"检测到导入日志 logs/{patient_id} 与现有患者编号冲突。\n\n"
                f"是否覆盖/合并到现有患者 {patient_id}？\n"
                f"选择“否”将保存为新编号：{alias_id}"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice == QMessageBox.Yes:
            self.db.upsert_patient(
                self._import_patient_data(patient_id, patient_dir, patient_info, overwrite=True)
            )
            return patient_id, False

        self.db.upsert_patient(
            self._import_patient_data(alias_id, patient_dir, patient_info, original_id=patient_id)
        )
        return alias_id, True

    def _insert_imported_session(
        self,
        patient_id: str,
        source_path: Path,
        events: list[tuple[str, float | None, str]],
    ) -> bool:
        note = self._log_path_note(source_path)
        if self._session_exists_for_note(note):
            return False
        cursor = self.db.conn.execute(
            "INSERT INTO sessions (patient_id, experiment_date, epoch_count, paradigm_type, notes) "
            "VALUES (?, ?, ?, ?, ?);",
            (
                patient_id,
                self._date_from_log_path(source_path),
                self._epoch_count_from_events(events),
                self._paradigm_type_from_log_path(source_path),
                note,
            ),
        )
        session_id = cursor.lastrowid
        created_at = datetime.now().isoformat(timespec="seconds")
        for name, timestamp, marker in events:
            self.db.conn.execute(
                "INSERT INTO swallow_events "
                "(session_id, event_name, timestamp_sec, marker_label, created_at) "
                "VALUES (?, ?, ?, ?, ?);",
                (session_id, name, timestamp, marker, created_at),
            )
        self.db.conn.commit()
        return True

    def _sync_logs_to_patient_db(self) -> tuple[int, int]:
        logs_root = _PROJECT_ROOT / "logs"
        if not logs_root.is_dir():
            return 0, 0

        added_patients = 0
        added_sessions = 0
        for patient_dir in sorted(p for p in logs_root.iterdir() if p.is_dir()):
            patient_id = patient_dir.name.strip()
            if not patient_id or patient_id.startswith("."):
                continue

            meta_files = sorted(patient_dir.rglob("*_meta.json"))
            csv_logs = sorted(patient_dir.rglob("swallow_*_log_*.csv"))
            if not meta_files and not csv_logs:
                continue

            # Prefer recorder metadata when present. CSV logs without matching
            # recordings are imported as fallback sessions.
            pending_sources: list[tuple[Path, list[tuple[str, float | None, str]], dict]] = []
            if meta_files:
                for meta_path in meta_files:
                    if self._session_exists_for_note(self._log_path_note(meta_path)):
                        continue
                    labels_path = Path(str(meta_path).replace("_meta.json", "_labels.json"))
                    events = self._load_events_from_labels(labels_path)
                    patient_info = self._load_patient_info_from_meta(meta_path)
                    pending_sources.append((meta_path, events, patient_info))
            else:
                for csv_path in csv_logs:
                    if self._session_exists_for_note(self._log_path_note(csv_path)):
                        continue
                    events = self._load_events_from_csv_log(csv_path)
                    pending_sources.append((csv_path, events, {}))

            if not pending_sources:
                continue

            patient_info = next((info for _, _, info in pending_sources if info), {})
            target_patient_id, created_patient = self._resolve_import_patient_id(
                patient_id, patient_dir, patient_info
            )
            if created_patient:
                added_patients += 1

            for source_path, events, _ in pending_sources:
                if self._insert_imported_session(target_patient_id, source_path, events):
                    added_sessions += 1

        if added_patients or added_sessions:
            self._append_log(
                f"[DB] 已从 logs 同步导入 {added_patients} 名患者、{added_sessions} 条历史会话。"
            )
        return added_patients, added_sessions

    def refresh_patients(self):
        prev = self.selected_patient_id
        self._sync_logs_to_patient_db()
        patients = self.db.get_patients()
        self.patient_stats = self.db.get_patient_stats()

        selected_row = None
        was_blocked = self._patient_table.blockSignals(True)
        try:
            self._patient_table.setRowCount(len(patients))
            for idx, row in enumerate(patients):
                pid = row["patient_id"]
                stats = self.patient_stats.get(pid, {"session_count": 0, "last_date": "-"})
                # sqlite3.Row uses index access, not .get()
                h = row["height_cm"]
                w = row["weight_kg"]
                values = [
                    pid, row["name"] or "", row["gender"] or "",
                    "" if row["age"] is None else str(row["age"]),
                    "" if h is None or h == 0 else f"{h:.1f}",
                    "" if w is None or w == 0 else f"{w:.1f}",
                    row["dysphagia_level"] or "未评估",
                ]
                for ci, v in enumerate(values):
                    item = QTableWidgetItem(v)
                    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                    self._patient_table.setItem(idx, ci, item)
                if prev and pid == prev:
                    selected_row = idx
            if selected_row is not None:
                self._patient_table.selectRow(selected_row)
            else:
                self._patient_table.clearSelection()
        finally:
            self._patient_table.blockSignals(was_blocked)
        self._on_patient_selected()

    def _on_patient_selected(self):
        rows = self._patient_table.selectionModel().selectedRows() if self._patient_table.selectionModel() else []
        if not rows:
            self.selected_patient_id = None
            for lbl in [self._lbl_id, self._lbl_name, self._lbl_gender,
                         self._lbl_age, self._lbl_level, self._lbl_sessions, self._lbl_last]:
                lbl.setText("-")
            self._btn_history.setEnabled(False)
            # 重置范式评估页签
            self._paradigm_patient_label.setText("未选择患者，请在「患者管理」页选中患者。")
            self._btn_start_paradigm.setEnabled(False)
            return

        def _txt(r, c): return (self._patient_table.item(r, c).text().strip()
                                  if self._patient_table.item(r, c) else "")

        row = rows[0].row()
        pid = _txt(row, 0)
        self.selected_patient_id = pid

        self._lbl_id.setText(pid)
        self._lbl_name.setText(_txt(row, 1) or "-")
        self._lbl_gender.setText(_txt(row, 2) or "-")
        self._lbl_age.setText(_txt(row, 3) or "-")
        self._lbl_level.setText(_txt(row, 6) or "-")

        stats = self.patient_stats.get(pid, {"session_count": 0, "last_date": "-"})
        self._lbl_sessions.setText(str(stats["session_count"]))
        self._lbl_last.setText(stats["last_date"])
        self._btn_history.setEnabled(True)

        # 同步更新范式评估页签和调控评估页签
        patient_info_text = f"已选中患者：{pid} | {_txt(row, 1) or '-'} | 吞咽等级：{_txt(row, 6) or '未评估'}"
        self._paradigm_patient_label.setText(patient_info_text)
        self._btn_start_paradigm.setEnabled(True)
        self._btn_start_part2.setEnabled(True)
        self._cl_patient_label.setText(patient_info_text)
        self._btn_start_cl.setEnabled(True)

    def _setup_demo_mode(self):
        """Prepare the real GUI for no-hardware demonstration."""
        self.db.upsert_patient({
            "patient_id": "DEMO_001",
            "name": "演示患者",
            "gender": "男",
            "age": 65,
            "height_cm": 170.0,
            "weight_kg": 65.0,
            "dysphagia_level": "演示病例",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
        self.selected_patient_id = "DEMO_001"
        self.refresh_patients()
        for row in range(self._patient_table.rowCount()):
            item = self._patient_table.item(row, 0)
            if item and item.text().strip() == "DEMO_001":
                self._patient_table.selectRow(row)
                break
        self._wifi_ip.setText("demo")
        self._wifi_port.setValue(9000)
        self._append_log("[Demo] 已启用演示模式：使用隔离数据库和模拟16通道EEG/EMG/ECG数据源。")
        self.tabs.setCurrentWidget(self._eeg_tab)
        QTimer.singleShot(300, self._connect_wifi)
        if self.demo_run:
            QTimer.singleShot(1200, self._run_demo_target)

    def _run_demo_target(self):
        if self.demo_run == "paradigm1":
            self._start_demo_paradigm(kind="part1")
        elif self.demo_run == "paradigm2":
            self._start_demo_paradigm(kind="part2")
        elif self.demo_run == "control":
            self._start_demo_control()

    def _start_demo_paradigm(self, kind: str):
        """Start the real paradigm subprocess with demo-safe defaults."""
        if not self.selected_patient_id:
            return
        if self._paradigm_process and self._paradigm_process.state() != QProcess.NotRunning:
            return
        script = PARADIGM_SCRIPT if kind == "part1" else PARADIGM_PART2_SCRIPT
        if not script.exists():
            QMessageBox.critical(self, "错误", f"未找到范式脚本：{script}")
            return
        args = [
            str(script),
            "--patient-id", self.selected_patient_id,
            "--epoch-count", "1",
            "--experiment-date", datetime.now().isoformat(timespec="seconds"),
            "--debug",
        ]
        self._active_paradigm_kind = kind
        self._recording_subdir = "cupture_data_part2" if kind == "part2" else None
        self._paradigm_process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONPATH", str(_PROJECT_ROOT))
        self._paradigm_process.setProcessEnvironment(env)
        self._paradigm_process.setProgram(str(DEFAULT_PYTHON))
        self._paradigm_process.setArguments(args)
        self._paradigm_process.readyReadStandardOutput.connect(self._on_paradigm_stdout)
        self._paradigm_process.readyReadStandardError.connect(self._on_paradigm_stderr)
        if kind == "part2":
            self._paradigm_process.started.connect(self._on_part2_started)
            self._paradigm_process.finished.connect(self._on_part2_finished)
        else:
            self._paradigm_process.started.connect(self._on_paradigm_started)
            self._paradigm_process.finished.connect(self._on_paradigm_finished)
        self._btn_start_paradigm.setEnabled(False)
        self._btn_start_part2.setEnabled(False)
        self._btn_stop_paradigm.setEnabled(True)
        self._paradigm_progress.setVisible(True)
        self._paradigm_progress.setRange(0, 0)
        self._append_log(f"[Demo] 启动真实范式脚本: python {' '.join(args)}")
        self.tabs.setCurrentWidget(self._log_tab)
        self._paradigm_process.start()

    def _start_demo_control(self):
        """Start the real control paradigm subprocess with demo defaults."""
        if not self.selected_patient_id:
            return
        if self._paradigm_process and self._paradigm_process.state() != QProcess.NotRunning:
            return
        opts = {
            "controller_host": "192.168.4.1",
            "controller_port": 3333,
            "controller_target": "ALL",
            "controller_duration": 1.0,
        }
        if self._eeg_amplifier is not None and self._eeg_amplifier.is_streaming():
            self._start_online_control(opts)
        else:
            self._append_log("[Demo] 未检测到模拟实时脑电，调控demo仅播放范式提示。")
        args = [
            str(CONTROL_PARADIGM_SCRIPT),
            "--patient-id", self.selected_patient_id,
            "--epoch-count", "1",
            "--experiment-date", datetime.now().isoformat(timespec="seconds"),
            "--debug",
            "--external-online-controller",
            "--controller-host", "192.168.4.1",
            "--controller-port", "3333",
            "--controller-target", "ALL",
            "--controller-duration", "1.0",
        ]
        self._paradigm_process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONPATH", str(_PROJECT_ROOT))
        self._paradigm_process.setProcessEnvironment(env)
        self._paradigm_process.setProgram(str(DEFAULT_PYTHON))
        self._paradigm_process.setArguments(args)
        self._paradigm_process.readyReadStandardOutput.connect(self._on_control_stdout)
        self._paradigm_process.readyReadStandardError.connect(self._on_paradigm_stderr)
        self._paradigm_process.started.connect(self._on_cl_started)
        self._paradigm_process.finished.connect(self._on_cl_finished)
        self._btn_start_cl.setEnabled(False)
        self._btn_stop_cl.setEnabled(True)
        self._cl_progress.setVisible(True)
        self._cl_progress.setRange(0, 0)
        self._append_log(f"[Demo] 启动真实调控范式脚本: python {' '.join(args)}")
        self.tabs.setCurrentWidget(self._log_tab)
        self._paradigm_process.start()

    def _selected_patient_info(self) -> Optional[dict]:
        if not self.selected_patient_id:
            return None
        def _cell(row: int, col: int) -> str:
            item = self._patient_table.item(row, col)
            return item.text().strip() if item else ""

        row_idx = self._patient_table.selectionModel().selectedRows()[0].row()
        return {
            "patient_id": _cell(row_idx, 0),
            "name": _cell(row_idx, 1),
            "gender": _cell(row_idx, 2),
            "age": _cell(row_idx, 3),
            "height_cm": _cell(row_idx, 4),
            "weight_kg": _cell(row_idx, 5),
            "dysphagia_level": _cell(row_idx, 6),
        }

    def _add_patient(self):
        dlg = PatientEditDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        try:
            self.db.upsert_patient(dlg.result())
            self._append_log("[DB] 患者已添加。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：{e}")
        self.refresh_patients()

    def _edit_patient(self):
        info = self._selected_patient_info()
        if not info:
            QMessageBox.warning(self, "提示", "请先选中一个患者。")
            return
        dlg = PatientEditDialog(existing=info, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        try:
            self.db.upsert_patient(dlg.result())
            self._append_log("[DB] 患者已更新。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"更新失败：{e}")
        self.refresh_patients()

    def _delete_patient(self):
        if not self.selected_patient_id:
            return
        ret = QMessageBox.question(self, "确认删除",
                                   f"确定要删除患者 {self.selected_patient_id} 及其所有测试数据吗？",
                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        self.db.delete_patient(self.selected_patient_id)
        self._append_log(f"[DB] 患者 {self.selected_patient_id} 已删除。")
        self.refresh_patients()

    def _show_history(self):
        if not self.selected_patient_id:
            return
        sessions = self.db.get_sessions(self.selected_patient_id)
        dlg = HistoryDialog(self.selected_patient_id, sessions, parent=self)
        dlg.exec()

    # ================================================================
    # Tab 2: 范式评估
    # ================================================================

    def _build_paradigm_tab(self):
        self._paradigm_tab = QWidget()
        root = QVBoxLayout(self._paradigm_tab)

        # 当前患者
        patient_grp = QGroupBox("当前测试对象")
        patient_layout = QVBoxLayout(patient_grp)
        self._paradigm_patient_label = QLabel("未选择患者，请在「患者管理」页选中患者。")
        patient_layout.addWidget(self._paradigm_patient_label)
        root.addWidget(patient_grp)

        # 范式流程预览
        flow_grp = QGroupBox("范式流程")
        flow_layout = QVBoxLayout(flow_grp)
        flow_label = QLabel(
            "静息(5s) → 想象吞咽1(5s) → 静息(5s) →\\n"
            "含水1→温水吞咽1(15s) → 实验结束"
        )
        flow_label.setStyleSheet("font-family: Microsoft YaHei; font-size: 13px; padding: 8px;")
        flow_layout.addWidget(flow_label)
        root.addWidget(flow_grp)

        # 执行控制
        action_grp = QGroupBox("执行控制")
        action_layout = QHBoxLayout(action_grp)
        self._btn_start_paradigm = QPushButton("▶ 开始范式测试")
        self._btn_start_paradigm.setMinimumHeight(40)
        self._btn_start_paradigm.setStyleSheet("font-size: 14px; font-weight: bold;")
        self._btn_stop_paradigm = QPushButton("■ 停止测试")
        self._btn_stop_paradigm.setMinimumHeight(40)
        self._btn_stop_paradigm.setEnabled(False)
        self._paradigm_status = QLabel("状态：空闲")
        self._paradigm_progress = QProgressBar()
        self._paradigm_progress.setVisible(False)

        action_layout.addWidget(self._btn_start_paradigm)
        action_layout.addWidget(self._btn_stop_paradigm)
        action_layout.addWidget(self._paradigm_status)
        root.addWidget(action_grp)
        root.addWidget(self._paradigm_progress)

        # 第二范式：静息 + 想象吞咽，用于part2采集和分类
        part2_flow_grp = QGroupBox("第二范式流程")
        part2_flow_layout = QVBoxLayout(part2_flow_grp)
        part2_flow_label = QLabel(
            "静息(5s) → 想象吞咽(5s) → 重复指定Epoch数量 → "
            "保存到 logs/<患者>/cupture_data_part2 → 自动调用分类算法"
        )
        part2_flow_label.setStyleSheet("font-family: Microsoft YaHei; font-size: 13px; padding: 8px;")
        part2_flow_layout.addWidget(part2_flow_label)
        root.addWidget(part2_flow_grp)

        part2_action_grp = QGroupBox("第二范式执行控制")
        part2_action_layout = QHBoxLayout(part2_action_grp)
        self._btn_start_part2 = QPushButton("▶ 开始第二范式采集")
        self._btn_start_part2.setMinimumHeight(40)
        self._btn_start_part2.setStyleSheet("font-size: 14px; font-weight: bold;")
        self._part2_status = QLabel("状态：空闲")
        part2_action_layout.addWidget(self._btn_start_part2)
        part2_action_layout.addWidget(self._part2_status)
        root.addWidget(part2_action_grp)

        root.addStretch(1)

        self._btn_start_paradigm.clicked.connect(self._start_paradigm)
        self._btn_stop_paradigm.clicked.connect(self._stop_paradigm)
        self._btn_start_part2.clicked.connect(self._start_part2_paradigm)
        self._btn_start_paradigm.setEnabled(False)
        self._btn_start_part2.setEnabled(False)

    def _start_paradigm(self):
        if not self.selected_patient_id:
            QMessageBox.warning(self, "提示", "请先在「患者管理」页选中一个患者。")
            return
        if self._paradigm_process and self._paradigm_process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "提示", "范式已在运行中。")
            return

        dlg = StartParadigmDialog(self.selected_patient_id, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.result()

        if not PARADIGM_SCRIPT.exists():
            QMessageBox.critical(self, "错误", f"未找到范式脚本：{PARADIGM_SCRIPT}")
            return

        args = [
            str(PARADIGM_SCRIPT),
            "--patient-id", self.selected_patient_id,
            "--epoch-count", str(opts["epoch_count"]),
            "--experiment-date", datetime.now().isoformat(timespec="seconds"),
        ]
        if opts["enable_lsl"]:
            args.extend(["--eeg-marker", "--marker-stream-name", opts["lsl_stream_name"]])
        if opts["debug"]:
            args.append("--debug")

        self._active_paradigm_kind = "part1"
        self._recording_subdir = None
        self._paradigm_process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONPATH", str(_PROJECT_ROOT))
        self._paradigm_process.setProcessEnvironment(env)
        self._paradigm_process.setProgram(str(DEFAULT_PYTHON))
        self._paradigm_process.setArguments(args)
        self._paradigm_process.readyReadStandardOutput.connect(self._on_paradigm_stdout)
        self._paradigm_process.readyReadStandardError.connect(self._on_paradigm_stderr)
        self._paradigm_process.started.connect(self._on_paradigm_started)
        self._paradigm_process.finished.connect(self._on_paradigm_finished)

        self._btn_start_paradigm.setEnabled(False)
        self._btn_stop_paradigm.setEnabled(True)
        self._paradigm_progress.setVisible(True)
        self._paradigm_progress.setRange(0, 0)
        self._append_log(f"[范式] 启动: python {' '.join(args)}")
        self.tabs.setCurrentWidget(self._log_tab)
        self._paradigm_process.start()

    def _start_part2_paradigm(self):
        if not self.selected_patient_id:
            QMessageBox.warning(self, "提示", "请先在「患者管理」页选中一个患者。")
            return
        if self._paradigm_process and self._paradigm_process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "提示", "范式已在运行中。")
            return

        dlg = StartParadigmDialog(
            self.selected_patient_id,
            parent=self,
            title="开始第二范式采集",
            epoch_default=10,
            lsl_default="Swallow_Part2_Markers",
        )
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.result()

        if not PARADIGM_PART2_SCRIPT.exists():
            QMessageBox.critical(self, "错误", f"未找到第二范式脚本：{PARADIGM_PART2_SCRIPT}")
            return

        args = [
            str(PARADIGM_PART2_SCRIPT),
            "--patient-id", self.selected_patient_id,
            "--epoch-count", str(opts["epoch_count"]),
            "--experiment-date", datetime.now().isoformat(timespec="seconds"),
        ]
        if opts["enable_lsl"]:
            args.extend(["--eeg-marker", "--marker-stream-name", opts["lsl_stream_name"]])
        if opts["debug"]:
            args.append("--debug")

        self._active_paradigm_kind = "part2"
        self._recording_subdir = "cupture_data_part2"
        self._paradigm_process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONPATH", str(_PROJECT_ROOT))
        self._paradigm_process.setProcessEnvironment(env)
        self._paradigm_process.setProgram(str(DEFAULT_PYTHON))
        self._paradigm_process.setArguments(args)
        self._paradigm_process.readyReadStandardOutput.connect(self._on_paradigm_stdout)
        self._paradigm_process.readyReadStandardError.connect(self._on_paradigm_stderr)
        self._paradigm_process.started.connect(self._on_part2_started)
        self._paradigm_process.finished.connect(self._on_part2_finished)

        self._btn_start_part2.setEnabled(False)
        self._btn_start_paradigm.setEnabled(False)
        self._btn_stop_paradigm.setEnabled(True)
        self._paradigm_progress.setVisible(True)
        self._paradigm_progress.setRange(0, 0)
        self._part2_status.setText("状态：运行中")
        self._append_log(f"[第二范式] 启动: python {' '.join(args)}")
        self.tabs.setCurrentWidget(self._log_tab)
        self._paradigm_process.start()

    def _on_paradigm_stdout(self):
        if not self._paradigm_process: return
        raw = self._paradigm_process.readAllStandardOutput()
        if not raw.size():
            return
        text = str(raw, "utf-8", errors="replace").strip()
        if not text: return
        for line in text.splitlines():
            line = line.strip()
            if not line: continue
            self._append_log(f"[范式] {line}")
            if line.startswith("AUDIO:"):
                self._play_audio(line[6:].strip())
                continue
            if line.startswith("EVENT:"):
                label = line[6:].strip()
                for key, (code, _) in self.PARADIGM_LABEL_MAP.items():
                    if key in label:
                        self._on_label_triggered(code, label, time.time())
                        break
                continue
            for key, (code, name) in self.PARADIGM_LABEL_MAP.items():
                if key in line:
                    self._on_label_triggered(code, name, time.time())
                    break

    def _on_paradigm_stderr(self):
        if not self._paradigm_process: return
        raw = self._paradigm_process.readAllStandardError()
        if not raw.size():
            return
        text = str(raw, "utf-8", errors="replace").strip()
        if text:
            for line in text.splitlines():
                line = line.strip()
                if line and "[Audio" not in line and "WARNING" not in line and "INFO" not in line:
                    self._append_log(f"[范式][ERR] {line}")

    def _play_audio(self, filename: str):
        """Play audio file in GUI process (has desktop access for MCI)."""
        filepath = _METABCI_DIR / "brainstim" / "assets" / filename
        if not filepath.is_file():
            filepath = Path(filename)
        if not filepath.is_file():
            return
        import ctypes
        try:
            winmm = ctypes.windll.winmm
            winmm.mciSendStringW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_void_p]
            winmm.mciSendStringW.restype = ctypes.c_uint
            f = str(filepath)
            alias = f"gui_{int(time.time()*1000)}"
            ret = winmm.mciSendStringW(f'open "{f}" alias {alias}', None, 0, None)
            if ret != 0:
                ret = winmm.mciSendStringW(f'open "{f}" type MPEGVideo alias {alias}', None, 0, None)
            if ret == 0:
                winmm.mciSendStringW(f'play {alias}', None, 0, None)
        except Exception:
            pass

    def _on_paradigm_started(self):
        self._paradigm_status.setText("状态：运行中")
        if self._eeg_amplifier is not None and self._eeg_amplifier.is_streaming():
            self._start_recording()
            self._append_log("[范式] 自动开始 EEG 录制")

    def _stop_paradigm(self):
        if self._paradigm_process and self._paradigm_process.state() != QProcess.NotRunning:
            self._paradigm_process.terminate()
            if not self._paradigm_process.waitForFinished(2500):
                self._paradigm_process.kill()
            self._append_log("[范式] 手动停止。")

    def _on_paradigm_finished(self, exit_code, _status):
        self._paradigm_status.setText(f"状态：已结束 (exit={exit_code})")
        self._paradigm_progress.setVisible(False)
        self._btn_start_paradigm.setEnabled(True)
        self._btn_start_part2.setEnabled(True)
        self._btn_stop_paradigm.setEnabled(False)
        self._append_log(f"[范式] 进程结束，退出码：{exit_code}")
        recording_paths = None
        if self._eeg_recorder and self._eeg_recorder.is_recording():
            self._stop_recording()
            recording_paths = getattr(self, "_last_recording_paths", None)
            self._append_log("[范式] 自动停止 EEG 录制")
        if self._paradigm_process:
            self._paradigm_process.deleteLater()
            self._paradigm_process = None
        self.refresh_patients()
        if self.demo_mode:
            QTimer.singleShot(0, self._setup_demo_mode)
        if exit_code == 0 and self.selected_patient_id:
            self._run_assessment()
            if recording_paths:
                self._run_part1_warm_prior_quantification(recording_paths)
        self._active_paradigm_kind = None
        self._recording_subdir = None

    def _on_part2_started(self):
        self._part2_status.setText("状态：运行中")
        if self._eeg_amplifier is not None and self._eeg_amplifier.is_streaming():
            self._start_recording()
            self._append_log("[第二范式] 自动开始 EEG/EMG/ECG 录制")
        else:
            self._append_log("[第二范式] 未检测到实时脑电连接，仅运行范式提示，不会生成分类输入数据。")

    def _on_part2_finished(self, exit_code, _status):
        self._part2_status.setText(f"状态：已结束 (exit={exit_code})")
        self._paradigm_progress.setVisible(False)
        self._btn_start_paradigm.setEnabled(True)
        self._btn_start_part2.setEnabled(True)
        self._btn_stop_paradigm.setEnabled(False)
        self._append_log(f"[第二范式] 进程结束，退出码：{exit_code}")

        recording_paths = None
        if self._eeg_recorder and self._eeg_recorder.is_recording():
            self._stop_recording()
            recording_paths = getattr(self, "_last_recording_paths", None)
            self._append_log("[第二范式] 自动停止 EEG/EMG/ECG 录制")

        if self._paradigm_process:
            self._paradigm_process.deleteLater()
            self._paradigm_process = None

        self.refresh_patients()
        if exit_code == 0 and self.selected_patient_id and recording_paths:
            self._run_part2_classification(recording_paths)
        elif exit_code == 0:
            self._append_log("[第二范式] 没有录制文件，跳过分类算法。")

        self._active_paradigm_kind = None
        self._recording_subdir = None

    def _run_part2_classification(self, recording_paths: dict):
        self._append_log("[第二范式] 正在调用竞赛分类算法...")
        try:
            result = run_part2_classification(
                recording_paths=recording_paths,
                patient_id=self.selected_patient_id or "",
                project_root=_PROJECT_ROOT,
            )
        except Exception as exc:
            self._append_log(f"[第二范式] 分类算法调用失败: {exc}")
            return

        status = result.get("status", "unknown")
        if status == "success":
            summary = result.get("prediction_summary", {}) or {}
            self._append_log(
                "[第二范式] 分类完成: "
                f"样本数={summary.get('n_predicted', '-')}, "
                f"ACC={summary.get('acc', '-')}, "
                f"BACC={summary.get('bacc', '-')}, "
                f"结果目录={result.get('output_dir')}"
            )
        elif status == "skipped":
            self._append_log(
                "[第二范式] 分类已跳过: 未找到训练好的 final_model.pt。"
                "请将模型放到 applications/swallow_bci/models/swallow_classifier/final_model.pt，"
                "或设置 METABCI_SWALLOW_CLASSIFIER_MODEL。"
            )
            self._append_log(f"[第二范式] 跳过原因文件: {result.get('output_dir')}")
        else:
            self._append_log(
                f"[第二范式] 分类失败: {result.get('error', 'unknown error')}，"
                f"详情见 {result.get('output_dir')}"
            )

    def _run_part1_warm_prior_quantification(self, recording_paths: dict):
        self._append_log("[范式] 正在调用竞赛量化算法（想象吞咽 + 温水吞咽）...")
        try:
            result = run_warm_prior_quantification(
                patient_id=self.selected_patient_id or "",
                recording_paths=recording_paths,
            )
        except Exception as exc:
            self._append_log(f"[范式] 竞赛量化算法调用失败: {exc}")
            return

        status = result.get("status", "unknown")
        if status == "success":
            self._append_log(
                "[范式] 竞赛量化完成: "
                f"target={result.get('target_score_1point', '-')}, "
                f"fitted={result.get('fitted_score_1point', '-')}, "
                f"loss={result.get('fit_loss', '-')}, "
                f"结果={result.get('fit_score_csv')}"
            )
        else:
            self._append_log(
                f"[范式] 竞赛量化失败: {result.get('error', 'unknown error')}，"
                f"详情见 {result.get('output_dir')}"
            )

    # 范式事件 → 标签编码映射
    PARADIGM_LABEL_MAP = {
        "静息": (1, "静息"),
        "想象吞咽": (2, "想象吞咽"),
        "含水": (3, "含水"),
        "温水吞咽": (4, "温水吞咽"),
        "实验开始": (5, "实验开始"),
        "实验结束": (6, "实验结束"),
    }

    def _find_latest_paradigm_log(self) -> Optional[str]:
        """查找指定患者最近的范式 CSV 日志文件。"""
        if not self.selected_patient_id:
            return None
        log_dir = Path(_PROJECT_ROOT) / "logs" / self.selected_patient_id
        if not log_dir.is_dir():
            return None
        csv_files = sorted(log_dir.glob("swallow_paradigm_log_*.csv"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        return str(csv_files[0]) if csv_files else None

    def _build_epochs_from_recording(
        self, npy_path: str, labels_path: str | None,
        meta_path: str | None,
    ) -> dict | None:
        """从录制文件构建 epochs 数据供 brainda 评估引擎使用。

        加载 NPY → 按标签时间戳分段 → 分离 EEG/EMG 通道 →
        返回 ``{"eeg": ndarray, "emg": ndarray}``。
        """
        try:
            data = np.load(npy_path)  # (n_channels, n_samples)
        except Exception:
            return None

        # Load metadata for srate + channel layout
        srate = 500.0
        eeg_ch = list(EEG_CHANNELS)
        emg_ch = list(EMG_CHANNELS)
        if meta_path and os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                srate = float(meta.get("srate", 500.0))
                eeg_ch = meta.get("eeg_channels", eeg_ch)
                emg_ch = meta.get("emg_channels", emg_ch)
            except Exception:
                pass

        # Load labels for epoch boundaries. Use both swallow events:
        # imagined swallow uses a 5 s window; warm-water swallow uses a 15 s window.
        event_windows = []
        if labels_path and os.path.isfile(labels_path):
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                for lb in labels:
                    name = str(lb.get("name", ""))
                    timestamp = float(lb.get("timestamp_sec", 0))
                    if "想象吞咽" in name:
                        event_windows.append((timestamp, 5.0))
                    elif "温水吞咽" in name:
                        event_windows.append((timestamp, 15.0))
            except Exception:
                pass

        if not event_windows:
            return None  # no swallow events to epoch around

        max_samples_epoch = int(max(duration for _, duration in event_windows) * srate)
        ecg_ch = list(ECG_CHANNELS)
        epochs_eeg, epochs_emg, epochs_ecg = [], [], []

        def _pad_epoch(seg: np.ndarray) -> np.ndarray:
            if seg.shape[1] >= max_samples_epoch:
                return seg[:, :max_samples_epoch]
            pad_width = max_samples_epoch - seg.shape[1]
            return np.pad(seg, ((0, 0), (0, pad_width)), mode="constant")

        for lt, duration in event_windows:
            start = max(0, int(lt * srate))
            n_samples_epoch = int(duration * srate)
            end = min(data.shape[1], start + n_samples_epoch)
            if end - start < n_samples_epoch:
                continue
            seg = data[:, start:end]  # (ch, time)
            eeg_seg = _pad_epoch(seg[[c for c in eeg_ch if c < seg.shape[0]], :])
            emg_seg = _pad_epoch(seg[[c for c in emg_ch if c < seg.shape[0]], :])
            ecg_seg = _pad_epoch(seg[[c for c in ecg_ch if c < seg.shape[0]], :])
            if eeg_seg.shape[0] > 0:
                epochs_eeg.append(eeg_seg)
            if emg_seg.shape[0] > 0:
                epochs_emg.append(emg_seg)
            if ecg_seg.shape[0] > 0:
                epochs_ecg.append(ecg_seg)
        if not epochs_eeg and not epochs_emg:
            return None

        result = {}
        if epochs_eeg:
            result["eeg"] = np.stack(epochs_eeg, axis=0)
        if epochs_emg:
            result["emg"] = np.stack(epochs_emg, axis=0)
        if epochs_ecg:
            result["ecg"] = np.stack(epochs_ecg, axis=0)
        return result

    def _run_assessment(self):
        """运行量化评估并显示报告弹窗。

        优先使用录制的 NPY 数据 → brainda 算法计算真实指标；
        回退到 CSV 日志 → 事件计数估算。
        """
        if not self.selected_patient_id:
            return

        csv_path = self._find_latest_paradigm_log()
        if not csv_path:
            self._append_log("[评估] 未找到范式日志，跳过自动评估。")
            return

        # Try loading recorded NPY data for brainda-based analysis
        epochs_data = None
        rec_paths = getattr(self, "_last_recording_paths", None) or {}
        npy_path = rec_paths.get("npy")
        labels_path = rec_paths.get("labels")
        meta_path = rec_paths.get("meta")

        if npy_path and os.path.isfile(npy_path):
            try:
                epochs_data = self._build_epochs_from_recording(
                    npy_path, labels_path, meta_path)
                if epochs_data:
                    self._append_log(
                        f"[评估] 已加载录制数据: {npy_path} → "
                        f"eeg={epochs_data.get('eeg', np.array([])).shape}, "
                        f"emg={epochs_data.get('emg', np.array([])).shape}, "
                        f"ecg={epochs_data.get('ecg', np.array([])).shape}")
            except Exception as e:
                self._append_log(f"[评估] 录制数据加载失败，使用CSV回退: {e}")

        self._append_log(f"[评估] 正在分析: {csv_path}")
        try:
            # Try to find a pre-trained model checkpoint
            model_path = _PROJECT_ROOT / "models" / "final_quantification_model.pt"
            model_path = str(model_path) if model_path.is_file() else None

            report = assess_from_paradigm_log(
                patient_id=self.selected_patient_id,
                csv_log_path=csv_path,
                epochs_data=epochs_data,
                model_path=model_path,
            )
        except Exception as e:
            self._append_log(f"[评估] 评估引擎异常: {e}")
            return

        self._append_log(
            f"[评估] 综合评分: {report['composite_score']}/100 "
            f"({report['composite_level']}), "
            f"建议阈值: {report['recommendations']['confidence_threshold']:.2f}"
        )

        # 显示评估报告弹窗
        dlg = AssessmentReportDialog(
            patient_id=self.selected_patient_id,
            report=report,
            parent=self,
        )
        dlg.exec()

    # ================================================================
    # Tab 3: 调控评估
    # ================================================================

    def _build_closed_loop_tab(self):
        self._closed_loop_tab = QWidget()
        root = QVBoxLayout(self._closed_loop_tab)

        # 当前测试对象
        patient_grp = QGroupBox("当前测试对象")
        patient_layout = QVBoxLayout(patient_grp)
        self._cl_patient_label = QLabel("未选择患者，请在「患者管理」页选中患者。")
        patient_layout.addWidget(self._cl_patient_label)
        root.addWidget(patient_grp)

        # 评估流程
        flow_grp = QGroupBox("调控范式流程")
        flow_layout = QVBoxLayout(flow_grp)
        flow_label = QLabel(
            "想象吞咽(5s) → 算法识别吞咽意图 → 电刺激触发点 →\n"
            "进入下一轮 Epoch，重复上述过程"
        )
        flow_label.setStyleSheet("font-family: Microsoft YaHei; font-size: 13px; padding: 8px;")
        flow_layout.addWidget(flow_label)
        root.addWidget(flow_grp)

        # 执行控制
        action_grp = QGroupBox("执行控制")
        action_layout = QHBoxLayout(action_grp)
        self._btn_start_cl = QPushButton("▶ 开始调控评估")
        self._btn_start_cl.setMinimumHeight(40)
        self._btn_start_cl.setStyleSheet("font-size: 14px; font-weight: bold;")
        self._btn_stop_cl = QPushButton("■ 停止评估")
        self._btn_stop_cl.setMinimumHeight(40)
        self._btn_stop_cl.setEnabled(False)
        self._btn_controller_gui = QPushButton("打开控制器页面")
        self._btn_controller_gui.setMinimumHeight(40)
        self._cl_status = QLabel("状态：空闲")
        self._cl_progress = QProgressBar()
        self._cl_progress.setVisible(False)

        action_layout.addWidget(self._btn_start_cl)
        action_layout.addWidget(self._btn_stop_cl)
        action_layout.addWidget(self._btn_controller_gui)
        action_layout.addWidget(self._cl_status)
        root.addWidget(action_grp)
        root.addWidget(self._cl_progress)
        root.addStretch(1)

        # 信号连接
        self._btn_start_cl.clicked.connect(self._start_cl_assessment)
        self._btn_stop_cl.clicked.connect(self._stop_cl_assessment)
        self._btn_controller_gui.clicked.connect(self._open_controller_gui)
        self._btn_start_cl.setEnabled(False)

    def _start_cl_assessment(self):
        """启动吞咽意图调控范式（QProcess + 音频中继到 GUI）。"""
        if not self.selected_patient_id:
            QMessageBox.warning(self, "提示", "请先在「患者管理」页选中一个患者。")
            return
        if self._paradigm_process and self._paradigm_process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "提示", "范式正在运行中。")
            return

        dlg = StartParadigmDialog(
            self.selected_patient_id,
            parent=self,
            title="开始调控范式",
            epoch_default=5,
            lsl_default="Swallow_Control_Markers",
            controller_settings=True,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        opts = dlg.result()

        if opts["enable_controller"]:
            if not self._eeg_amplifier or not self._eeg_amplifier.is_streaming():
                QMessageBox.warning(
                    self,
                    "提示",
                    "请先在「实时脑电」页连接设备，再启用自动电刺激闭环。",
                )
                return
            if not self._start_online_control(opts):
                return
        else:
            self._stop_online_control()

        if not CONTROL_PARADIGM_SCRIPT.exists():
            QMessageBox.critical(self, "错误", f"未找到调控范式脚本：{CONTROL_PARADIGM_SCRIPT}")
            return

        args = [
            str(CONTROL_PARADIGM_SCRIPT),
            "--patient-id", self.selected_patient_id,
            "--epoch-count", str(opts["epoch_count"]),
            "--experiment-date", datetime.now().isoformat(timespec="seconds"),
        ]
        if opts["enable_lsl"]:
            args.extend(["--eeg-marker", "--marker-stream-name", opts["lsl_stream_name"]])
        if opts["debug"]:
            args.append("--debug")
        if opts["enable_controller"]:
            args.append("--external-online-controller")
        args.extend([
            "--controller-host", opts["controller_host"],
            "--controller-port", str(opts["controller_port"]),
            "--controller-target", opts["controller_target"],
            "--controller-duration", str(opts["controller_duration"]),
        ])

        self._paradigm_process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONPATH", str(_PROJECT_ROOT))
        self._paradigm_process.setProcessEnvironment(env)
        self._paradigm_process.setProgram(str(DEFAULT_PYTHON))
        self._paradigm_process.setArguments(args)
        self._paradigm_process.readyReadStandardOutput.connect(self._on_control_stdout)
        self._paradigm_process.readyReadStandardError.connect(self._on_paradigm_stderr)
        self._paradigm_process.started.connect(self._on_cl_started)
        self._paradigm_process.finished.connect(self._on_cl_finished)

        self._btn_start_cl.setEnabled(False)
        self._btn_stop_cl.setEnabled(True)
        self._cl_progress.setVisible(True)
        self._cl_progress.setRange(0, 0)
        self._append_log(f"[调控评估] 启动: python {' '.join(args)}")
        self.tabs.setCurrentWidget(self._log_tab)
        self._paradigm_process.start()

    def _start_online_control(self, opts: dict) -> bool:
        """Enable online EEG-window classification and ESP32B triggering."""
        try:
            srate = self._eeg_amplifier.get_srate() if self._eeg_amplifier else 500.0
            detector = OnlineSwallowIntentDetector(
                project_root=_PROJECT_ROOT,
                srate=srate,
                threshold=0.6,
                device="cpu",
            )
            controller = Esp32UdpController(
                host=opts["controller_host"],
                port=opts["controller_port"],
            )
        except Exception as exc:
            QMessageBox.warning(self, "调控闭环启动失败", str(exc))
            self._append_log(f"[在线调控] 启动失败: {exc}")
            return False

        token = object()
        self._online_control_enabled = True
        self._online_control_token = token
        self._online_detector = detector
        self._online_controller = controller
        self._online_pending_windows = []
        self._online_inference_busy = False
        self._online_trigger_busy = False
        self._online_control_settings = {
            "target": opts["controller_target"],
            "duration": float(opts["controller_duration"]),
            "threshold": float(detector.threshold),
            "token": token,
        }
        self._online_result_queue = queue.Queue()
        self._append_log(
            "[在线调控] 已启用: "
            f"mode={detector.mode}, threshold={detector.threshold:.2f}, "
            f"window={detector.window_samples / max(float(srate), 1.0):.1f}s, "
            f"ESP32B={opts['controller_host']}:{opts['controller_port']}, "
            f"target={opts['controller_target']}, duration={float(opts['controller_duration']):.1f}s"
        )
        if detector.error:
            self._append_log(f"[在线调控] {detector.error}")
        return True

    def _stop_online_control(self):
        self._online_control_enabled = False
        self._online_control_token = None
        self._online_detector = None
        self._online_controller = None
        self._online_control_settings = {}
        self._online_pending_windows = []
        self._online_inference_busy = False
        self._online_trigger_busy = False
        self._online_result_queue = queue.Queue()

    def _queue_online_intent_window(self, label: str):
        if not self._online_control_enabled or self._online_detector is None:
            return
        if not self._eeg_amplifier or not self._eeg_amplifier.is_streaming():
            self._append_log("[在线调控] 未连接实时脑电，跳过本次闭环判别。")
            return
        start_sample = self._current_eeg_sample_count()
        window_samples = int(self._online_detector.window_samples)
        event = {
            "label": label,
            "start_sample": int(start_sample),
            "end_sample": int(start_sample + window_samples),
            "window_samples": window_samples,
            "token": self._online_control_token,
            "queued_at": time.time(),
        }
        self._online_pending_windows.append(event)
        self._append_log(
            "[在线调控] 已锁定想象吞咽窗口: "
            f"{label}, samples={event['start_sample']}..{event['end_sample']}"
        )

    def _poll_online_control(self):
        self._drain_online_result_queue()
        if (not self._online_control_enabled
                or self._online_detector is None
                or self._online_inference_busy
                or not self._online_pending_windows):
            return

        event = self._online_pending_windows[0]
        if event.get("token") is not self._online_control_token:
            self._online_pending_windows.pop(0)
            return
        current_sample = self._current_eeg_sample_count()
        if current_sample < int(event["end_sample"]):
            return

        window = self._get_eeg_live_window(
            int(event["start_sample"]),
            int(event["window_samples"]),
        )
        self._online_pending_windows.pop(0)
        if window is None:
            self._append_log("[在线调控] 实时缓存不足，无法截取本次5秒窗口。")
            return

        self._online_inference_busy = True
        threading.Thread(
            target=self._run_online_inference_worker,
            args=(dict(event), window),
            daemon=True,
        ).start()

    def _run_online_inference_worker(self, event: dict, window: np.ndarray):
        try:
            result = self._online_detector.predict(window) if self._online_detector else {
                "detected": False,
                "confidence": 0.0,
                "mode": "detector_missing",
                "threshold": 1.0,
                "label": "非吞咽意图",
            }
            self._online_result_queue.put(("inference", event, result))
        except Exception as exc:
            self._online_result_queue.put(("inference_error", event, repr(exc)))

    def _run_online_trigger_worker(self, event: dict, result: dict):
        try:
            if self._online_controller is None:
                raise RuntimeError("ESP32B controller is not initialized")
            target = self._online_control_settings.get("target", "ALL")
            duration = float(self._online_control_settings.get("duration", 1.0))
            response = self._online_controller.trigger(target=target, duration=duration)
            self._online_result_queue.put(("trigger", event, result, response))
        except Exception as exc:
            self._online_result_queue.put(("trigger_error", event, result, repr(exc)))

    def _drain_online_result_queue(self):
        while True:
            try:
                item = self._online_result_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "inference":
                _, event, result = item
                self._online_inference_busy = False
                self._handle_online_inference_result(event, result)
            elif kind == "inference_error":
                _, event, error = item
                self._online_inference_busy = False
                self._append_log(
                    f"[在线调控] 分类失败: {event.get('label', '-')}, {error}"
                )
            elif kind == "trigger":
                _, event, result, response = item
                self._online_trigger_busy = False
                self._append_log(
                    "[在线调控] ESP32B 电刺激已触发: "
                    f"{event.get('label', '-')}, confidence={float(result.get('confidence', 0.0)):.3f}, "
                    f"response={response}"
                )
            elif kind == "trigger_error":
                _, event, result, error = item
                self._online_trigger_busy = False
                self._append_log(
                    "[在线调控] ESP32B 电刺激触发失败: "
                    f"{event.get('label', '-')}, confidence={float(result.get('confidence', 0.0)):.3f}, "
                    f"{error}"
                )

    def _handle_online_inference_result(self, event: dict, result: dict):
        if event.get("token") is not self._online_control_token:
            return
        confidence = float(result.get("confidence", 0.0))
        threshold = float(result.get("threshold", self._online_control_settings.get("threshold", 0.6)))
        detected = bool(result.get("detected", False)) and confidence >= threshold
        self._append_log(
            "[在线调控] 分类结果: "
            f"{event.get('label', '-')}, mode={result.get('mode', '-')}, "
            f"confidence={confidence:.3f}, threshold={threshold:.3f}, "
            f"detected={int(detected)}"
        )
        if not detected:
            self._append_log("[在线调控] 未达到触发阈值，本轮不发送电刺激。")
            return
        if self._online_trigger_busy:
            self._append_log("[在线调控] 上一次电刺激命令仍在执行，本轮触发已跳过。")
            return
        self._online_trigger_busy = True
        threading.Thread(
            target=self._run_online_trigger_worker,
            args=(dict(event), dict(result)),
            daemon=True,
        ).start()

    def _on_control_stdout(self):
        if not self._paradigm_process:
            return
        raw = self._paradigm_process.readAllStandardOutput()
        if not raw.size():
            return
        text = str(raw, "utf-8", errors="replace").strip()
        if not text:
            return
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            self._append_log(f"[调控评估] {line}")
            if line.startswith("AUDIO:"):
                self._play_audio(line[6:].strip())
                continue
            if line.startswith("EVENT:"):
                label = line[6:].strip()
                for key, (code, _) in self.PARADIGM_LABEL_MAP.items():
                    if key in label:
                        self._on_label_triggered(code, label, time.time())
                        break
                if self._online_control_enabled and "想象吞咽" in label:
                    self._queue_online_intent_window(label)
                continue
            if line.startswith("INTENT:"):
                if "detected=1" in line:
                    if self._online_control_enabled:
                        self._append_log("[调控评估] 范式已进入想象吞咽提示，等待实时分类闭环结果。")
                    else:
                        self._append_log("[调控评估] 算法识别到想象吞咽意图。")
                continue
            if line.startswith("CONTROL_PLACEHOLDER:"):
                if self._online_control_enabled:
                    self._append_log("[调控评估] 子进程预留触发已忽略；真实电刺激由实时分类闭环控制。")
                else:
                    self._append_log("[调控评估] 控制器预留接口已到达，未启用真实硬件触发。")
                continue
            if line.startswith("CONTROL_TRIGGER:"):
                self._append_log("[调控评估] ESP32B 控制器已收到自动触发命令。")
                continue
            if line.startswith("CONTROL_ERROR:"):
                self._append_log(f"[调控评估] ESP32B 控制器触发失败：{line[14:].strip()}")
                continue

    def _open_controller_gui(self):
        if not CONTROLLER_GUI_SCRIPT.exists():
            QMessageBox.critical(self, "错误", f"未找到控制器脚本：{CONTROLLER_GUI_SCRIPT}")
            return
        if self._controller_process and self._controller_process.state() != QProcess.NotRunning:
            self.tabs.setCurrentWidget(self._log_tab)
            self._append_log("[控制器] 控制器页面已经在运行。")
            return
        self._controller_process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONPATH", str(_PROJECT_ROOT))
        self._controller_process.setProcessEnvironment(env)
        self._controller_process.setProgram(str(DEFAULT_PYTHON))
        self._controller_process.setArguments([str(CONTROLLER_GUI_SCRIPT)])
        self._controller_process.readyReadStandardOutput.connect(self._read_controller_stdout)
        self._controller_process.readyReadStandardError.connect(self._read_controller_stderr)
        self._controller_process.finished.connect(self._on_controller_finished)
        self._append_log(f"[控制器] 启动: python {CONTROLLER_GUI_SCRIPT}")
        self._controller_process.start()

    def _read_controller_stdout(self):
        if not self._controller_process:
            return
        text = self._decode_process_output(bytes(self._controller_process.readAllStandardOutput())).strip()
        if text:
            self._append_log(f"[控制器] {text}")

    def _read_controller_stderr(self):
        if not self._controller_process:
            return
        text = self._decode_process_output(bytes(self._controller_process.readAllStandardError())).strip()
        if text:
            self._append_log(f"[控制器][ERR] {text}")

    def _on_controller_finished(self, exit_code, _status):
        self._append_log(f"[控制器] 页面已关闭 (exit={exit_code})")
        if self._controller_process:
            self._controller_process.deleteLater()
            self._controller_process = None

    def _on_cl_started(self):
        self._cl_status.setText("状态：运行中")
        if self._eeg_amplifier is not None and self._eeg_amplifier.is_streaming():
            self._start_recording()
            self._append_log("[调控评估] 自动开始 EEG 录制")

    def _stop_cl_assessment(self):
        self._stop_online_control()
        if self._paradigm_process and self._paradigm_process.state() != QProcess.NotRunning:
            self._paradigm_process.terminate()
            self._append_log("[调控评估] 手动停止。")

    def _on_cl_finished(self, exit_code, _status):
        self._cl_status.setText(f"状态：已结束 (exit={exit_code})")
        self._cl_progress.setVisible(False)
        self._btn_start_cl.setEnabled(True)
        self._btn_stop_cl.setEnabled(False)
        self._append_log(f"[调控评估] 进程结束，退出码：{exit_code}")
        if self._eeg_recorder and self._eeg_recorder.is_recording():
            self._stop_recording()
            self._append_log("[调控评估] 自动停止 EEG 录制")
        if self._paradigm_process:
            self._paradigm_process.deleteLater()
            self._paradigm_process = None
        self.refresh_patients()
        self._stop_online_control()
        # 调控范式不自动生成吞咽功能评估报告。

    def _get_cl_config(self) -> dict:
        return {
            "decoder": "riemann" if self._radio_riemann.isChecked() else "eegnet",
            "threshold": self._threshold_slider.value() / 100.0,
            "stim_type": self._stim_combo.currentText(),
            "stim_duration": self._stim_duration.value(),
            "mode": ("sim" if self._radio_sim.isChecked()
                      else "real" if self._radio_real.isChecked()
                      else "wifi"),
            "duration": self._duration_spin.value(),
            "n_channels": self._ch_spin.value(),
            "stream_name": self._lsl_stream_edit.text().strip() or "OpenBCI_GUI",
        }

    def _train_decoder(self):
        cfg = self._get_cl_config()
        args = [
            str(CLOSED_LOOP_SCRIPT),
            "--mode", "sim",
            "--decoder", cfg["decoder"],
            "--n-channels", str(cfg["n_channels"]),
            "--duration", str(cfg["duration"]),
            "--threshold", str(cfg["threshold"]),
            "--stim", cfg["stim_type"],
        ]
        self._run_cl_process(args, "训练解码器并运行")

    def _run_closed_loop(self):
        cfg = self._get_cl_config()
        args = [
            str(CLOSED_LOOP_SCRIPT),
            "--mode", cfg["mode"],
            "--decoder", cfg["decoder"],
            "--n-channels", str(cfg["n_channels"]),
            "--duration", str(cfg["duration"]),
            "--threshold", str(cfg["threshold"]),
            "--stim", cfg["stim_type"],
        ]
        if cfg["mode"] == "real":
            args.extend(["--stream-name", cfg["stream_name"]])
        elif cfg["mode"] == "wifi":
            args.extend([
                "--host", self._wifi_ip.text().strip() or "192.168.4.1",
                "--port", str(self._wifi_port.value()),
            ])
        self._run_cl_process(args, "开始调控评估")

    def _run_cl_process(self, args, action_name):
        self._closed_loop_process = QProcess(self)
        # 强制 UTF-8 + 为子进程注入项目根路径（解压即用）
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONPATH", str(_PROJECT_ROOT))
        self._closed_loop_process.setProcessEnvironment(env)
        self._closed_loop_process.setProgram(str(DEFAULT_PYTHON))
        self._closed_loop_process.setArguments(args)
        self._closed_loop_process.readyReadStandardOutput.connect(self._read_cl_stdout)
        self._closed_loop_process.readyReadStandardError.connect(self._read_cl_stderr)
        self._closed_loop_process.finished.connect(self._on_cl_finished)

        self._cl_status.setText("状态：运行中")
        self._cl_progress.setRange(0, 0)
        self._append_log(f"[调控评估] {action_name}: python {' '.join(args)}")
        self.tabs.setCurrentWidget(self._log_tab)
        self._closed_loop_process.start()

    def _stop_closed_loop(self):
        if self._closed_loop_process and self._closed_loop_process.state() != QProcess.NotRunning:
            self._closed_loop_process.terminate()

    def _read_cl_stdout(self):
        if not self._closed_loop_process: return
        raw = bytes(self._closed_loop_process.readAllStandardOutput())
        text = self._decode_process_output(raw).strip()
        if text:
            self._append_log(f"[调控评估] {text}")
            # 尝试解析反馈数据
            for line in text.split("\n"):
                if "已处理:" in line:
                    import re
                    chunks = re.search(r'已处理:\s*(\d+)', line)
                    detected = re.search(r'检测:\s*(\d+)', line)
                    triggers = re.search(r'触发:\s*(\d+)', line)
                    conf = re.search(r'置信度:\s*([\d.]+)', line)
                    if chunks: self._fb_chunks.setText(chunks.group(1))
                    if detected: self._fb_detected.setText(detected.group(1))
                    if triggers: self._fb_triggers.setText(triggers.group(1))
                    if conf: self._fb_conf.setText(conf.group(1))

    def _read_cl_stderr(self):
        if not self._closed_loop_process: return
        raw = bytes(self._closed_loop_process.readAllStandardError())
        text = self._decode_process_output(raw).strip()
        if text:
            self._append_log(f"[调控评估][ERR] {text}")

    def _save_model(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存模型", "", "Model Files (*.pkl *.pt)")
        if not path:
            return
        cfg = self._get_cl_config()
        self._run_cl_process([
            str(CLOSED_LOOP_SCRIPT),
            "--mode", "sim", "--decoder", cfg["decoder"],
            "--n-channels", str(cfg["n_channels"]),
            "--duration", "30", "--save-model", path,
        ], "训练并保存模型")

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "加载模型", "", "Model Files (*.pkl *.pt)")
        if not path:
            return
        cfg = self._get_cl_config()
        self._run_cl_process([
            str(CLOSED_LOOP_SCRIPT),
            "--mode", cfg["mode"], "--decoder", cfg["decoder"],
            "--n-channels", str(cfg["n_channels"]),
            "--duration", str(cfg["duration"]),
            "--threshold", str(cfg["threshold"]),
            "--stim", cfg["stim_type"],
            "--load-model", path,
        ], "加载模型并运行")

    # ================================================================
    # Tab 4: 实时脑电（brainflow LSL 数据源）
    # ================================================================

    def _build_eeg_tab(self):
        self._eeg_tab = QWidget()
        root = QVBoxLayout(self._eeg_tab)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        # ---- 连接栏（单行紧凑）----
        conn_layout = QHBoxLayout()
        conn_layout.addWidget(QLabel("IP:"))
        self._wifi_ip = QLineEdit("192.168.4.1")
        self._wifi_ip.setMaximumWidth(110)
        self._wifi_ip.setFixedHeight(28)
        conn_layout.addWidget(self._wifi_ip)
        conn_layout.addWidget(QLabel("端口:"))
        self._wifi_port = QSpinBox()
        self._wifi_port.setRange(1024, 65535)
        self._wifi_port.setValue(9000)
        self._wifi_port.setMaximumWidth(75)
        self._wifi_port.setFixedHeight(28)
        conn_layout.addWidget(self._wifi_port)
        self._btn_connect = QPushButton("连接")
        self._btn_connect.setFixedHeight(28)
        self._btn_disconnect = QPushButton("断开")
        self._btn_disconnect.setFixedHeight(28)
        self._btn_disconnect.setEnabled(False)
        conn_layout.addWidget(self._btn_connect)
        conn_layout.addWidget(self._btn_disconnect)
        self._conn_status = QLabel("未连接")
        conn_layout.addWidget(self._conn_status)
        self._eeg_status = QLabel("")
        conn_layout.addWidget(self._eeg_status)
        conn_layout.addStretch(1)
        conn_layout.addWidget(QLabel("滚轮调窗 | "))
        conn_layout.addWidget(QLabel(
            f"EEG:{len(EEG_CHANNELS)} EMG:{len(EMG_CHANNELS)} "
            f"ECG:{len(ECG_CHANNELS)}"))
        root.addLayout(conn_layout)

        # ---- 波形显示（最大化）----
        self._eeg_plot = None
        self._plot_container = QWidget()
        self._plot_layout = QVBoxLayout(self._plot_container)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        placeholder = QLabel("点击「连接」开始实时波形显示")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet(
            "color: #aaa; font-size: 16px; padding: 60px;"
        )
        self._plot_layout.addWidget(placeholder)
        root.addWidget(self._plot_container, stretch=1)

        # ---- 信号连接 ----
        self._btn_connect.clicked.connect(self._connect_wifi)
        self._btn_disconnect.clicked.connect(self._disconnect_wifi)

    # ================================================================
    # EEG Tab 事件处理
    # ================================================================

    def _on_ch_config_changed(self):
        pass  # 通道已固定分配，无需联动

    def _connect_wifi(self):
        """连接 WiFi Shield 并自动开始采集+录制。"""
        if not HAS_PYQTGRAPH:
            QMessageBox.critical(
                self, "依赖缺失", "pyqtgraph 未安装。pip install pyqtgraph"
            )
            return

        ip = self._wifi_ip.text().strip()
        port = self._wifi_port.value()
        total_ch = 16
        srate = 500.0
        use_demo_source = self.demo_mode or ip.lower() in {"demo", "sim", "simulation"}

        endpoint = "DemoSwallowAmplifier" if use_demo_source else f"{ip}:{port}"
        self._append_log(f"[EEG] 连接 {endpoint} ...")
        self._eeg_status.setText(f"连接 {endpoint} ...")
        self._btn_connect.setEnabled(False)

        if use_demo_source:
            amplifier = DemoSwallowAmplifier(n_channels=total_ch, srate=srate)
        else:
            amplifier = WiFiShieldAmplifier(
                host=ip, port=port, n_channels=total_ch, srate=srate,
            )

        self._start_eeg_connect_worker(amplifier, {"ip": ip, "endpoint": endpoint})

    def _start_eeg_connect_worker(self, amplifier, context: dict):
        token = object()
        self._eeg_connect_token = token
        self._eeg_connect_result = None
        self._eeg_pending_amplifier = amplifier
        self._eeg_pending_context = dict(context)

        def _worker():
            error = None
            try:
                amplifier.start()
                if token is not self._eeg_connect_token:
                    try:
                        amplifier.stop()
                    except Exception:
                        pass
            except Exception as exc:
                error = exc
            self._eeg_connect_result = (token, amplifier, error)

        self._eeg_connect_thread = threading.Thread(
            target=_worker, name="EEGConnectWorker", daemon=True,
        )
        self._eeg_connect_thread.start()

        self._eeg_connect_timer = QTimer(self)
        self._eeg_connect_timer.timeout.connect(self._poll_eeg_connect_result)
        self._eeg_connect_timer.start(100)

        self._eeg_connect_timeout_timer = QTimer(self)
        self._eeg_connect_timeout_timer.setSingleShot(True)
        self._eeg_connect_timeout_timer.timeout.connect(
            lambda tok=token: self._on_eeg_connect_timeout(tok)
        )
        self._eeg_connect_timeout_timer.start(10000)

    def _poll_eeg_connect_result(self):
        result = self._eeg_connect_result
        if result is None:
            return

        token, amplifier, error = result
        self._eeg_connect_result = None
        if token is not self._eeg_connect_token:
            if error is None:
                threading.Thread(target=amplifier.stop, daemon=True).start()
            return

        self._stop_eeg_connect_timers()
        self._eeg_connect_token = None
        self._eeg_pending_amplifier = None
        context = dict(self._eeg_pending_context)
        self._eeg_pending_context = {}

        if error is not None:
            QMessageBox.warning(self, "连接失败", str(error))
            self._append_log(f"[EEG] 失败: {error}")
            self._eeg_status.setText("失败")
            self._conn_status.setText("未连接")
            self._conn_status.setStyleSheet("")
            self._btn_connect.setEnabled(True)
            return

        self._finish_eeg_connection(amplifier, context)

    def _on_eeg_connect_timeout(self, token: object):
        if token is not self._eeg_connect_token:
            return

        amplifier = self._eeg_pending_amplifier
        self._eeg_connect_token = None
        self._eeg_pending_amplifier = None
        self._eeg_pending_context = {}
        self._stop_eeg_connect_timers()
        if amplifier is not None:
            threading.Thread(target=amplifier.stop, daemon=True).start()

        self._append_log("[EEG] 连接超时：10秒内未搜索到设备。")
        self._eeg_status.setText("连接超时")
        self._conn_status.setText("未连接")
        self._conn_status.setStyleSheet("")
        self._btn_connect.setEnabled(True)
        QMessageBox.warning(self, "连接超时", "需要连接设备")

    def _stop_eeg_connect_timers(self):
        if self._eeg_connect_timer is not None:
            self._eeg_connect_timer.stop()
            self._eeg_connect_timer = None
        if self._eeg_connect_timeout_timer is not None:
            self._eeg_connect_timeout_timer.stop()
            self._eeg_connect_timeout_timer = None

    def _finish_eeg_connection(self, amplifier, context: dict):
        self._eeg_amplifier = amplifier
        self._eeg_live_buffer = None
        self._eeg_live_buffer_start_sample = 0
        self._eeg_live_last_end_sample = 0
        ip = context.get("ip", "")
        actual_ch = amplifier.get_n_channels()
        actual_srate = amplifier.get_srate()
        self._append_log(f"[EEG] 已连接 {actual_ch}ch {actual_srate:.0f}Hz")
        self._conn_status.setText(f"● {ip}")
        self._conn_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self._btn_connect.setEnabled(False)
        self._btn_disconnect.setEnabled(True)

        # 波形显示
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        self._eeg_plot = MultiRegionEEGWidget(
            eeg_channels=EEG_CHANNELS,
            emg_channels=EMG_CHANNELS,
            ecg_channels=ECG_CHANNELS,
            srate=actual_srate,
            visible_duration=8.0,
        )
        self._plot_layout.addWidget(self._eeg_plot)

        # 轮询定时器
        self._eeg_plot_timer = QTimer(self)
        self._eeg_plot_timer.timeout.connect(self._update_eeg_plot)
        self._eeg_plot_timer.start(33)

        self._eeg_status.setText(
            f"{actual_ch}ch@{actual_srate:.0f}Hz"
        )

    def _disconnect_wifi(self):
        """断开设备。"""
        self._stop_online_control()
        if self._eeg_plot_timer:
            self._eeg_plot_timer.stop()
            self._eeg_plot_timer = None

        # 停止录制
        if self._eeg_recorder and self._eeg_recorder.is_recording():
            self._stop_recording()

        if self._eeg_amplifier:
            try:
                self._eeg_amplifier.stop()
            except Exception as e:
                self._append_log(f"[EEG] 停止异常: {e}")
            self._eeg_amplifier = None

        # 清理波形
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        p = QLabel("点击「连接」开始")
        p.setAlignment(Qt.AlignCenter)
        p.setStyleSheet("color: #aaa; font-size: 16px; padding: 60px;")
        self._plot_layout.addWidget(p)
        self._eeg_plot = None
        self._eeg_live_buffer = None
        self._eeg_live_buffer_start_sample = 0
        self._eeg_live_last_end_sample = 0

        self._conn_status.setText("未连接")
        self._conn_status.setStyleSheet("")
        self._btn_connect.setEnabled(True)
        self._btn_disconnect.setEnabled(False)
        self._eeg_status.setText("")
        self._append_log("[EEG] 已断开。")

    def _update_eeg_plot(self):
        """QTimer 回调：刷新波形 + 追加录制数据。"""
        if self._eeg_amplifier is None or not self._eeg_amplifier.is_streaming():
            return

        chunk_size = int(self._eeg_amplifier.get_srate() * 0.1)
        new_data = self._eeg_amplifier.get_recent(chunk_size)

        if new_data is not None and self._eeg_plot is not None:
            self._eeg_plot.update_data(new_data)
        if new_data is not None:
            self._append_eeg_live_buffer(new_data)

        if (self._eeg_recorder is not None
                and self._eeg_recorder.is_recording()
                and new_data is not None):
            self._eeg_recorder.append_data(new_data)

    def _current_eeg_sample_count(self) -> int:
        if self._eeg_amplifier is not None and hasattr(self._eeg_amplifier, "get_sample_count"):
            try:
                return int(self._eeg_amplifier.get_sample_count())
            except Exception:
                pass
        if self._eeg_live_buffer is not None:
            return int(self._eeg_live_buffer_start_sample + self._eeg_live_buffer.shape[1])
        return int(self._eeg_live_last_end_sample)

    def _append_eeg_live_buffer(self, data: np.ndarray):
        chunk = np.asarray(data, dtype=np.float32)
        if chunk.ndim != 2 or chunk.size == 0:
            return
        expected_channels = (
            self._eeg_amplifier.get_n_channels()
            if self._eeg_amplifier is not None else chunk.shape[0]
        )
        if chunk.shape[0] != expected_channels and chunk.shape[1] == expected_channels:
            chunk = chunk.T
        n_channels, n_samples = chunk.shape
        if n_samples <= 0:
            return

        end_sample = self._current_eeg_sample_count()
        if end_sample < self._eeg_live_last_end_sample + n_samples:
            end_sample = self._eeg_live_last_end_sample + n_samples
        start_sample = end_sample - n_samples

        if self._eeg_live_buffer is None:
            self._eeg_live_buffer = chunk.copy()
            self._eeg_live_buffer_start_sample = start_sample
        else:
            if self._eeg_live_buffer.shape[0] != n_channels:
                self._eeg_live_buffer = chunk.copy()
                self._eeg_live_buffer_start_sample = start_sample
            else:
                self._eeg_live_buffer = np.concatenate([self._eeg_live_buffer, chunk], axis=1)

        self._eeg_live_last_end_sample = end_sample
        srate = self._eeg_amplifier.get_srate() if self._eeg_amplifier else 500.0
        max_samples = max(1, int(float(srate) * self._eeg_live_max_seconds))
        if self._eeg_live_buffer.shape[1] > max_samples:
            drop = self._eeg_live_buffer.shape[1] - max_samples
            self._eeg_live_buffer = self._eeg_live_buffer[:, drop:]
            self._eeg_live_buffer_start_sample += drop

    def _get_eeg_live_window(self, start_sample: int, n_samples: int) -> Optional[np.ndarray]:
        if self._eeg_live_buffer is None:
            return None
        rel_start = int(start_sample) - int(self._eeg_live_buffer_start_sample)
        rel_end = rel_start + int(n_samples)
        if rel_start < 0 or rel_end > self._eeg_live_buffer.shape[1]:
            return None
        return self._eeg_live_buffer[:, rel_start:rel_end].copy()

    def _current_patient_info_for_meta(self) -> dict:
        if not self.selected_patient_id:
            return {}
        row = self.db.get_patient(self.selected_patient_id)
        if row is None:
            return {"patient_id": self.selected_patient_id}
        return {
            "patient_id": row["patient_id"],
            "name": row["name"],
            "gender": row["gender"],
            "age": row["age"],
            "height_cm": row["height_cm"],
            "weight_kg": row["weight_kg"],
            "dysphagia_level": row["dysphagia_level"],
        }

    def _start_recording(self):
        """开始录制 EEG/EMG 数据到文件。"""
        if self._eeg_amplifier is None:
            QMessageBox.warning(self, "提示", "请先连接设备。")
            return

        subject = self.selected_patient_id or "subject01"
        log_dir = _PROJECT_ROOT / "logs" / subject
        if self._recording_subdir:
            log_dir = log_dir / self._recording_subdir
        srate = self._eeg_amplifier.get_srate()
        total_ch = self._eeg_amplifier.get_n_channels()
        labels = [
            ALL_CHANNEL_NAMES[i] if i < len(ALL_CHANNEL_NAMES)
            else f"Ch{i + 1}" for i in range(total_ch)
        ]

        self._eeg_recorder = EEGRecorder(
            output_dir=str(log_dir),
            subject_id=subject,
            srate=srate,
            n_channels=total_ch,
            channel_labels=labels,
            eeg_channels=list(EEG_CHANNELS),
            emg_channels=list(EMG_CHANNELS),
            patient_info=self._current_patient_info_for_meta(),
        )
        self._eeg_recorder.start_session()

        if hasattr(self, "_btn_record"):
            self._btn_record.setEnabled(False)
        if hasattr(self, "_btn_stop_record"):
            self._btn_stop_record.setEnabled(True)
        self._eeg_status.setText(
            f"录制中 — {self._eeg_status.text().split(' — ')[-1]}"
        )
        self._append_log(
            f"[EEG] 开始录制: 受试者={subject}, "
            f"{total_ch}ch, {srate:.0f}Hz → {log_dir}"
        )

    def _stop_recording(self):
        """停止录制并保存文件。"""
        if self._eeg_recorder is None:
            return
        self._eeg_recorder.stop_session()
        paths = self._eeg_recorder.get_file_paths()
        self._last_recording_paths = paths  # store for assessment
        self._append_log(
            f"[EEG] 录制完成: NPY={paths['npy']}, "
            f"标签数={len(self._eeg_recorder._labels)}"
        )
        self._eeg_recorder = None

    def _on_label_triggered(self, code: int, name: str, timestamp: float):
        """标签按钮回调：注入到放大器事件流 + 写入录制器。"""
        if self._eeg_amplifier is not None:
            self._eeg_amplifier.inject_label(code)

        if (self._eeg_recorder is not None
                and self._eeg_recorder.is_recording()):
            self._eeg_recorder.add_label(code, name)

        self._append_log(f"[标签] code={code}, name='{name}'")

    def _update_filters(self):
        pass  # 滤波已内置

    def _on_window_changed(self, text: str):
        if self._eeg_plot is not None:
            self._eeg_plot.set_visible_duration(float(text.replace("s", "")))

    # ================================================================
    # Tab 5: 日志
    # ================================================================

    def _build_log_tab(self):
        self._log_tab = QWidget()
        root = QVBoxLayout(self._log_tab)
        actions = QHBoxLayout()
        btn_clear = QPushButton("清空日志")
        btn_export = QPushButton("导出日志")
        actions.addWidget(btn_clear)
        actions.addWidget(btn_export)
        actions.addStretch(1)
        root.addLayout(actions)

        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(5000)
        self._log_text.setPlaceholderText("实验输出日志会显示在这里（brainflow get_logger 输出）...")
        root.addWidget(self._log_text)

        btn_clear.clicked.connect(self._log_text.clear)
        btn_export.clicked.connect(self._export_log)

        # 状态栏
        self._status_bar = QLabel("● 就绪 | DB: ✓ | brainflow: ✓")
        self._status_bar.setStyleSheet("padding: 4px; background: #2d2d2d; color: #ccc;")
        root.addWidget(self._status_bar)

    def _append_log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_text.appendPlainText(f"[{ts}] {message}")

    def _export_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出日志", "swallow_log.txt", "Text Files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._log_text.toPlainText())

    # ================================================================
    # 窗口关闭
    # ================================================================

    def closeEvent(self, event):
        self._stop_online_control()
        if self._online_control_timer:
            self._online_control_timer.stop()
        for proc in [self._paradigm_process, self._closed_loop_process, self._eeg_process, self._controller_process]:
            if proc and proc.state() != QProcess.NotRunning:
                proc.terminate()
                proc.waitForFinished(2000)
        # 停止 EEG 采集
        if self._eeg_plot_timer:
            self._eeg_plot_timer.stop()
        if self._eeg_recorder and self._eeg_recorder.is_recording():
            self._eeg_recorder.stop_session()
        if self._eeg_amplifier:
            try:
                self._eeg_amplifier.stop()
            except Exception:
                pass
        self.db.close()
        super().closeEvent(event)


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="脑卒中吞咽困难调控系统 GUI")
    parser.add_argument("--demo", action="store_true", help="启用无硬件演示模式")
    parser.add_argument(
        "--demo-run",
        choices=["", "paradigm1", "paradigm2", "control"],
        default="",
        help="演示模式下自动启动指定流程",
    )
    parser.add_argument(
        "--demo-auto-close",
        type=float,
        default=0.0,
        help="演示模式自动关闭秒数，仅用于自动化验证",
    )
    args, qt_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *qt_args]
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow(demo_mode=args.demo, demo_run=args.demo_run)
    window.show()
    if args.demo and args.demo_auto_close > 0:
        QTimer.singleShot(int(args.demo_auto_close * 1000), app.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
