# -*- coding: utf-8 -*-
"""脑卒中吞咽困难调控系统 — brainda 算法评估结果弹窗。

属于 metabci.brainstim.ui 包。
"""

from __future__ import annotations
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QGridLayout,
    QHBoxLayout,
)


class DecoderResultDialog(QDialog):
    """P300 解码器评估结果弹窗。

    显示由 brainda 算法（RiemannianMDM / EEGNet）评估后的性能指标。
    """

    def __init__(
        self,
        patient_id: str,
        result: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"解码器评估结果 — {patient_id}")
        self.setModal(True)
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)

        # 标题
        layout.addWidget(QLabel(f"患者：{patient_id}"))
        decoder_name = result.get("decoder_name", "未知")
        layout.addWidget(QLabel(f"解码器：{decoder_name}"))

        # 性能指标网格
        metrics_group = QGroupBox("brainda 算法评估指标")
        grid = QGridLayout(metrics_group)

        metrics = [
            ("准确率 (Accuracy)", result.get("accuracy"), "%"),
            ("平衡准确率 (bAcc)", result.get("balanced_accuracy"), "%"),
            ("理论 ITR", result.get("theoretical_itr"), "bits/min"),
            ("实际 ITR", result.get("practical_itr"), "bits/min"),
            ("真正率 (TPR)", result.get("tpr"), ""),
            ("假正率 (FPR)", result.get("fpr"), ""),
            ("AUC", result.get("auc"), ""),
        ]

        for i, (label, value, unit) in enumerate(metrics):
            row, col = i // 2, (i % 2) * 2
            grid.addWidget(QLabel(label), row, col)
            if value is not None:
                if isinstance(value, float):
                    display = f"{value:.2f} {unit}".strip()
                else:
                    display = str(value)
            else:
                display = "N/A"
            val_label = QLabel(display)
            val_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            grid.addWidget(val_label, row, col + 1)

        layout.addWidget(metrics_group)

        # 详细信息
        details_group = QGroupBox("详细信息")
        details_layout = QVBoxLayout(details_group)
        details = QTextEdit()
        details.setReadOnly(True)

        report_lines = [
            f"解码器类型: {decoder_name}",
            f"通道数: {result.get('n_channels', '?')}",
            f"采样率: {result.get('srate', '?')} Hz",
            f"训练试次: {result.get('n_train', '?')}",
            f"测试试次: {result.get('n_test', '?')}",
            f"混淆矩阵:\n{result.get('confusion_matrix', 'N/A')}",
            "",
            f"时间: {result.get('timestamp', '?')}",
        ]
        details.setPlainText("\n".join(report_lines))
        details.setMinimumHeight(180)
        details_layout.addWidget(details)
        layout.addWidget(details_group)

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class ClosedLoopResultDialog(QDialog):
    """闭环调控运行结果弹窗。"""

    def __init__(self, patient_id: str, summary: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"闭环调控结果 — {patient_id}")
        self.setModal(True)
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"患者：{patient_id}"))

        grid = QGridLayout()
        items = [
            ("运行模式", summary.get("mode", "-")),
            ("解码器", summary.get("decoder", "-")),
            ("置信度阈值", f"{summary.get('threshold', 0):.0%}"),
            ("总时长", f"{summary.get('elapsed', 0):.1f}s"),
            ("处理数据块", str(summary.get("total_chunks", 0))),
            ("检测意图次数", str(summary.get("detected_swallows", 0))),
            ("触发电刺激", str(summary.get("total_triggers", 0))),
        ]

        for i, (label, value) in enumerate(items):
            grid.addWidget(QLabel(label), i, 0)
            val_label = QLabel(value)
            val_label.setStyleSheet("font-weight: bold;")
            grid.addWidget(val_label, i, 1)

        layout.addLayout(grid)
        layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


# ============================================================
# 综合评估报告弹窗
# ============================================================

class AssessmentReportDialog(QDialog):
    """吞咽功能量化评估报告弹窗 — 仅显示综合评分。"""

    def __init__(
        self,
        patient_id: str,
        report: dict,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"吞咽功能评估 — {patient_id}")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title = QLabel("吞咽功能评估报告")
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        layout.addWidget(QLabel(
            f"患者: {patient_id}    日期: {report.get('timestamp', '-')}"))

        composite_score = report.get("composite_score", 0)
        composite_level = report.get("composite_level", "未知")

        score_label = QLabel(f"{composite_score}")
        score_label.setStyleSheet(
            f"font-size: 64px; font-weight: bold; "
            f"color: {self._score_color(composite_score)};")
        score_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(score_label)

        level_label = QLabel(f"等级: {composite_level}")
        level_label.setAlignment(Qt.AlignCenter)
        level_label.setStyleSheet("font-size: 18px;")
        layout.addWidget(level_label)

        comp_bar = QProgressBar()
        comp_bar.setRange(0, 100)
        comp_bar.setValue(composite_score)
        comp_bar.setTextVisible(True)
        comp_bar.setFormat(f"{composite_score}/100")
        comp_bar.setMinimumHeight(20)
        layout.addWidget(comp_bar)

        layout.addStretch()

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        btn_close.setMinimumHeight(32)
        layout.addWidget(btn_close)

    @staticmethod
    def _score_color(score: float) -> str:
        if score >= 85:
            return "#4CAF50"
        elif score >= 70:
            return "#2196F3"
        elif score >= 50:
            return "#FF9800"
        elif score >= 30:
            return "#F44336"
        else:
            return "#9E9E9E"
