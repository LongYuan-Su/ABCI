# -*- coding: utf-8 -*-
"""脑卒中吞咽困难调控系统 — 患者编辑和历史弹窗。

属于 metabci.brainflow.gui 包。
"""

from __future__ import annotations
from typing import Optional

import sqlite3
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QComboBox,
    QDoubleSpinBox,
)


# ============================================================
# 患者编辑弹窗
# ============================================================

class PatientEditDialog(QDialog):
    """录入 / 更新患者信息。"""

    def __init__(self, existing: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("录入 / 更新患者")
        self.setModal(True)
        self.setMinimumWidth(420)
        self._result: Optional[dict] = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.patient_id_edit = QLineEdit()
        self.patient_id_edit.setPlaceholderText("必填，如 P001")
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("患者姓名")

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["", "男", "女", "其他"])

        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 150)
        self.age_spin.setSpecialValueText("未填写")
        self.age_spin.setValue(0)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.0, 250.0)
        self.height_spin.setDecimals(1)
        self.height_spin.setSuffix(" cm")
        self.height_spin.setSpecialValueText("未填写")
        self.height_spin.setValue(0.0)

        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0.0, 300.0)
        self.weight_spin.setDecimals(1)
        self.weight_spin.setSuffix(" kg")
        self.weight_spin.setSpecialValueText("未填写")
        self.weight_spin.setValue(0.0)

        self.dysphagia_combo = QComboBox()
        self.dysphagia_combo.addItems([
            "未评估",
            "Ⅰ级(正常)",
            "Ⅱ级(可疑轻度)",
            "Ⅲ级(中度)",
            "Ⅳ级(中重度)",
            "Ⅴ级(重度)",
        ])

        form.addRow("患者ID *", self.patient_id_edit)
        form.addRow("姓名", self.name_edit)
        form.addRow("性别", self.gender_combo)
        form.addRow("年龄", self.age_spin)
        form.addRow("身高", self.height_spin)
        form.addRow("体重", self.weight_spin)
        form.addRow("吞咽障碍等级", self.dysphagia_combo)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # 填充已有数据 (兼容 dict 和 sqlite3.Row)
        if existing:
            # Normalise to dict for sqlite3.Row compatibility
            ex = dict(existing) if not isinstance(existing, dict) else existing
            self.patient_id_edit.setText(ex.get("patient_id", ""))
            self.patient_id_edit.setReadOnly(True)
            self.name_edit.setText(ex.get("name", ""))
            gender = ex.get("gender", "")
            if gender in ["男", "女", "其他"]:
                self.gender_combo.setCurrentText(gender)
            try:
                age = int(ex.get("age", 0) or 0)
                self.age_spin.setValue(age)
            except (ValueError, TypeError):
                pass
            try:
                height = float(ex.get("height_cm", 0) or 0)
                self.height_spin.setValue(height)
            except (ValueError, TypeError):
                pass
            try:
                weight = float(ex.get("weight_kg", 0) or 0)
                self.weight_spin.setValue(weight)
            except (ValueError, TypeError):
                pass
            level = ex.get("dysphagia_level", "未评估")
            if level and level != "未评估":
                idx = self.dysphagia_combo.findText(level)
                if idx >= 0:
                    self.dysphagia_combo.setCurrentIndex(idx)

    def _on_accept(self):
        patient_id = self.patient_id_edit.text().strip()
        if not patient_id:
            QMessageBox.warning(self, "提示", "患者ID不能为空。")
            return

        age_val = self.age_spin.value() if self.age_spin.value() > 0 else None
        height_val = self.height_spin.value() if self.height_spin.value() > 0 else None
        weight_val = self.weight_spin.value() if self.weight_spin.value() > 0 else None
        level = self.dysphagia_combo.currentText()
        if level == "未评估":
            level = None

        self._result = {
            "patient_id": patient_id,
            "name": self.name_edit.text().strip() or None,
            "gender": self.gender_combo.currentText().strip() or None,
            "age": age_val,
            "height_cm": height_val,
            "weight_kg": weight_val,
            "dysphagia_level": level,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.accept()

    def result(self) -> dict:
        if self._result is None:
            raise RuntimeError("result() 在对话框确认前不可用。")
        return self._result


# ============================================================
# 历史测试弹窗
# ============================================================

class HistoryDialog(QDialog):
    """查看患者的范式测试历史。"""

    def __init__(self, patient_id: str, sessions: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"测试历史 — {patient_id}")
        self.resize(820, 480)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"患者：{patient_id}，共 {len(sessions)} 次测试"))

        if not sessions:
            layout.addWidget(QLabel("暂无测试记录。"))
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(self.reject)
            buttons.accepted.connect(self.accept)
            layout.addWidget(buttons)
            return

        table = QTableWidget(len(sessions), 6)
        table.setHorizontalHeaderLabels([
            "日期", "会话ID", "Epoch次数", "范式类型", "备注", "事件数"
        ])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        def _value(record, key, default=""):
            try:
                value = record[key]
            except Exception:
                value = record.get(key, default) if hasattr(record, "get") else default
            return default if value is None else value

        for row_idx, s in enumerate(sessions):
            values = [
                _value(s, "experiment_date", "-") or "-",
                str(_value(s, "session_id", "-")),
                str(_value(s, "epoch_count", 0)),
                _value(s, "paradigm_type", "swallow_assessment") or "swallow_assessment",
                _value(s, "notes", "") or "-",
                str(_value(s, "event_count", 0)),
            ]
            for col_idx, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                table.setItem(row_idx, col_idx, item)

        layout.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
