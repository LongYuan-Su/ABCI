# -*- coding: utf-8 -*-
"""Launcher for the real GUI workflow demos."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "metabci" / "brainflow" / "gui" / "main_window.py").is_file():
            return parent
    raise RuntimeError("Cannot find MetaBCI project root from demo launcher path.")


PROJECT_ROOT = _find_project_root()
MAIN_WINDOW = PROJECT_ROOT / "metabci" / "brainflow" / "gui" / "main_window.py"


class Launcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("吞咽项目真实流程 Demo 启动器")
        self.resize(680, 360)
        self.processes: list[QProcess] = []

        root = QWidget()
        layout = QVBoxLayout(root)

        title = QLabel("选择要运行的真实流程 Demo")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font: bold 20px 'Microsoft YaHei'; padding: 12px; color: #1d4ed8;")
        layout.addWidget(title)

        for label, mode in [
            ("主程序 Demo：原 GUI + 模拟实时脑电", ""),
            ("范式一 Demo：原 GUI + 播放真实范式一", "paradigm1"),
            ("范式二 Demo：原 GUI + 播放真实范式二", "paradigm2"),
            ("调控评估 Demo：原 GUI + 播放真实调控范式", "control"),
        ]:
            button = QPushButton(label)
            button.setMinimumHeight(42)
            button.clicked.connect(lambda _=False, m=mode: self.launch(m))
            layout.addWidget(button)

        note = QLabel(
            "说明：页面来自原 main_window.py；数据源为 DemoSwallowAmplifier；"
            "范式脚本仍为 metabci/brainstim 下的真实脚本。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("font: 13px 'Microsoft YaHei'; color: #475569; padding: 8px;")
        layout.addWidget(note)
        self.setCentralWidget(root)

    def launch(self, mode: str):
        process = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONPATH", str(PROJECT_ROOT))
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(env)
        process.setProgram(sys.executable)
        args = [str(MAIN_WINDOW), "--demo"]
        if mode:
            args.extend(["--demo-run", mode])
        process.setArguments(args)
        self.processes.append(process)
        process.finished.connect(lambda *_: self.processes.remove(process) if process in self.processes else None)
        process.start()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    win = Launcher()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
