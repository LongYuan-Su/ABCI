#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令行脑电数据查看工具 — 基于 ``metabci.brainda`` 模块的 I/O 和可视化能力。

- 通道名称规范化使用 ``metabci.brainda.utils.channels.upper_ch_names``
- 数据加载对标 ``metabci.brainda.utils.io`` 模式
- 更丰富的 ERP 可视化参见 ``metabci.brainda.algorithms.feature_analysis.TimeAnalysis``
  （plot_single_trial / plot_multi_trials / plot_topomap）

用法:
  python view_data.py                          # 列出所有记录文件
  python view_data.py <文件路径>                # 查看文件信息
  python view_data.py <文件路径> --head 20      # 显示前20行
  python view_data.py <文件路径> --stats        # 显示统计信息
  python view_data.py <文件路径> --plot 1-4     # 绘制通道1-4的波形图
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))


def _get_recordings_dir() -> Path:
    """Cross-platform path to the recordings directory (uses pathlib)."""
    return Path(__file__).resolve().parent / "recordings"


def list_recordings(recording_dir: str | Path) -> None:
    """列出所有记录文件。"""
    rec_dir = Path(recording_dir)
    if not rec_dir.is_dir():
        print(f"记录目录不存在: {rec_dir}")
        return
    files = sorted(
        [f for f in rec_dir.iterdir() if f.suffix == ".csv"],
        reverse=True,
    )
    if not files:
        print("暂无记录文件。")
        return
    print(f"\n{'='*70}")
    print(f"  记录文件列表 ({rec_dir})")
    print(f"{'='*70}")
    for i, f in enumerate(files):
        size_kb = f.stat().st_size / 1024
        with open(f, "r", encoding="utf-8") as fh:
            n_lines = sum(1 for _ in fh) - 1
        print(f"  [{i+1}] {f.name}  |  {size_kb:.1f} KB  |  {n_lines} 采样点")
    print(f"{'='*70}\n")


def _try_normalise_channel_names(raw_names: list[str]) -> list[str]:
    """Normalise channel names using brainda's channel utilities.

    Delegates to ``metabci.brainda.utils.channels.upper_ch_names`` for
    standard 10-20 name formatting.  Falls back to the raw names if
    brainda is not importable.
    """
    try:
        from metabci.brainda.utils.channels import upper_ch_names
        return upper_ch_names(raw_names)
    except ImportError:
        return raw_names


def show_info(filepath: str | Path) -> None:
    """显示文件基本信息。"""
    fpath = Path(filepath)
    if not fpath.is_file():
        print(f"文件不存在: {fpath}")
        return
    with open(fpath, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        ch_names_raw = header.split(",")[1:]
        n_channels = len(ch_names_raw)
        lines = f.readlines()
        n_samples = len(lines)

    size_kb = fpath.stat().st_size / 1024
    first_ts = float(lines[0].split(",")[0]) if lines else 0
    last_ts = float(lines[-1].split(",")[0]) if lines else 0
    duration = last_ts - first_ts

    ch_names_display = _try_normalise_channel_names(ch_names_raw)

    print(f"\n{'='*60}")
    print(f"  文件: {fpath.name}")
    print(f"{'='*60}")
    print(f"  通道数:     {n_channels}")
    print(f"  通道名称:   {', '.join(ch_names_display[:16])}")
    if n_channels > 16:
        print(f"              ... (共 {n_channels} 通道)")
    print(f"  采样点数:   {n_samples}")
    print(f"  时长:       {duration:.2f} 秒")
    if duration > 0:
        print(f"  采样率:     {n_samples / duration:.1f} Hz")
    print(f"  文件大小:   {size_kb:.1f} KB")
    print(f"  起始时间:   {first_ts:.3f} s")
    print(f"  结束时间:   {last_ts:.3f} s")
    print(f"{'='*60}\n")


def show_head(filepath: str | Path, n: int = 10) -> None:
    """显示前N行数据。"""
    fpath = Path(filepath)
    if not fpath.is_file():
        print(f"文件不存在: {fpath}")
        return
    with open(fpath, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        print(f"\n{header}")
        print("-" * 80)
        for i, line in enumerate(f):
            if i >= n:
                break
            parts = line.strip().split(",")
            ts = float(parts[0])
            vals = [float(x) for x in parts[1:]]
            val_str = " ".join(f"{v:8.2f}" for v in vals[:8])
            if len(vals) > 8:
                val_str += " …"
            print(f"  t={ts:8.3f} | {val_str}")
    print()


def show_stats(filepath: str | Path) -> None:
    """显示各通道的统计信息。

    Uses ``np.loadtxt`` for fast tabular loading — same pattern as
    ``metabci.brainda.utils.io.loadmat`` for numeric data I/O.
    """
    fpath = Path(filepath)
    if not fpath.is_file():
        print(f"文件不存在: {fpath}")
        return

    data = np.loadtxt(str(fpath), delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    signals = data[:, 1:]

    with open(fpath, "r", encoding="utf-8") as f:
        ch_names = f.readline().strip().split(",")[1:]

    print(f"\n{'='*70}")
    print(f"  通道统计信息")
    print(f"{'='*70}")
    print(f"  {'通道':<10} {'均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10} {'范围':>10}")
    print(f"  {'-'*60}")

    for i, name in enumerate(ch_names):
        if i >= signals.shape[1]:
            break
        ch_data = signals[:, i]
        mean_val = np.mean(ch_data)
        std_val = np.std(ch_data)
        min_val = np.min(ch_data)
        max_val = np.max(ch_data)
        rng = max_val - min_val
        print(f"  {name:<10} {mean_val:10.2f} {std_val:10.2f} "
              f"{min_val:10.2f} {max_val:10.2f} {rng:10.2f}")
    print(f"{'='*70}\n")


def show_plot(filepath: str | Path, channels: str = "1-8") -> None:
    """使用 matplotlib 绘制通道波形。

    For richer ERP visualisation with amplitude markers and multi-trial
    overlays, see the brainda TimeAnalysis plotting API:
      ``metabci.brainda.algorithms.feature_analysis.TimeAnalysis``
        - ``plot_single_trial(data, sample_num, amp_mark=True)``
        - ``plot_multi_trials(data, sample_num)``
        - ``plot_topomap(data, point, channels, fig)``
    """
    fpath = Path(filepath)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return

    if not fpath.is_file():
        print(f"文件不存在: {fpath}")
        return

    ch_indices = []
    for part in channels.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            ch_indices.extend(range(int(a) - 1, int(b)))
        else:
            ch_indices.append(int(part) - 1)

    data = np.loadtxt(str(fpath), delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    timestamps = data[:, 0]
    signals = data[:, 1:]

    with open(fpath, "r", encoding="utf-8") as f:
        ch_names = f.readline().strip().split(",")[1:]

    n_ch = len(ch_indices)
    fig, axes = plt.subplots(n_ch, 1, figsize=(12, 2 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    fig.suptitle(f"脑电数据: {fpath.name}", fontsize=12)

    for ax, ch_idx in zip(axes, ch_indices):
        if ch_idx >= signals.shape[1]:
            continue
        name = ch_names[ch_idx] if ch_idx < len(ch_names) else f"Ch{ch_idx+1}"
        ax.plot(timestamps, signals[:, ch_idx], linewidth=0.5, color="#2e86c1")
        ax.set_ylabel(f"{name}\n(μV)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("时间 (秒)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    recording_dir = _get_recordings_dir()

    parser = argparse.ArgumentParser(
        description="命令行脑电数据查看工具 (基于 metabci.brainda 模块)"
    )
    parser.add_argument(
        "file", nargs="?",
        help="CSV 文件路径 (不指定则列出所有记录文件)"
    )
    parser.add_argument(
        "--head", type=int, default=0, metavar="N",
        help="显示前 N 行数据"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="显示各通道统计信息"
    )
    parser.add_argument(
        "--plot", type=str, default="", metavar="CHANNELS",
        help="绘制指定通道波形 (例如: 1-8 或 1,3,5)"
    )
    parser.add_argument(
        "--dir", type=str, default=str(recording_dir),
        help=f"记录文件目录 (默认: {recording_dir})"
    )

    args = parser.parse_args()

    if args.file is None:
        list_recordings(args.dir)
        return

    filepath = Path(args.file)
    if not filepath.is_absolute():
        filepath = Path(args.dir) / filepath

    if not filepath.is_file():
        alt_path = Path(args.dir) / Path(args.file).name
        if alt_path.is_file():
            filepath = alt_path
        else:
            print(f"文件不存在: {args.file}")
            print(f"查找目录: {args.dir}")
            list_recordings(args.dir)
            return

    show_info(str(filepath))

    if args.head > 0:
        show_head(str(filepath), args.head)

    if args.stats:
        show_stats(str(filepath))

    if args.plot:
        show_plot(str(filepath), args.plot)


if __name__ == "__main__":
    main()
