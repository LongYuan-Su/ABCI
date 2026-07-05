# -*- coding: utf-8 -*-
"""Launch the real integrated GUI in demo mode."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "metabci" / "brainflow" / "gui" / "main_window.py").is_file():
            return parent
    raise RuntimeError("Cannot find MetaBCI project root from demo launcher path.")


PROJECT_ROOT = _find_project_root()
MAIN_WINDOW = PROJECT_ROOT / "metabci" / "brainflow" / "gui" / "main_window.py"


def launch(demo_run: str = "", extra: list[str] | None = None) -> int:
    args = [
        sys.executable,
        str(MAIN_WINDOW),
        "--demo",
    ]
    if demo_run:
        args.extend(["--demo-run", demo_run])
    if extra:
        args.extend(extra)
    return subprocess.call(args, cwd=str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="启动原主程序的无硬件 demo 模式")
    parser.add_argument("--demo-run", choices=["", "paradigm1", "paradigm2", "control"], default="")
    parser.add_argument("--demo-auto-close", type=float, default=0.0)
    ns = parser.parse_args()
    extra = []
    if ns.demo_auto_close > 0:
        extra.extend(["--demo-auto-close", str(ns.demo_auto_close)])
    return launch(ns.demo_run, extra)


if __name__ == "__main__":
    raise SystemExit(main())
