# brainflow 目录说明

本目录保留 MetaBCI 原有在线采集框架，并承载本项目的实时采集、记录、处理、调控和 GUI 入口。

## 根目录保留的原 MetaBCI 核心

- `amplifiers.py`：放大器抽象、环形缓存、Marker、LSL 接入等基础能力。
- `workers.py`：在线处理 Worker。
- `logger.py`：统一日志接口。

这几个文件与原版 `MetaBCI-master/metabci/brainflow` 对齐，尽量不做结构性移动。

## 新增/扩展子包

- `acquisition/`：数据源、采集记录和运行时数据库。
  - `sources.py`：OpenBCI WiFi Shield、模拟数据源、实时缓冲区。
  - `recorder.py`：EEG/EMG/ECG 数据、marker、患者元数据保存。
  - `database.py`：患者、会话和事件 SQLite 数据库管理。
  - `real_time_eeg.py`：独立实时波形显示工具。
  - `view_data.py`：命令行数据查看工具。

- `processing/`：信号处理和评估算法。
  - `assessment.py`：吞咽评估结果计算。
  - `decoder.py`：解码器工厂。
  - `feature_extraction.py`：在线特征提取。
  - `signal_quality.py`：通道信号质量检测。

- `control/`：调控闭环和电刺激接口。
  - `closed_loop.py`：闭环调控脚本，可由主 GUI 子进程启动。
  - `online_swallow_control.py`：实时吞咽想象检测器。
  - `stimulator.py`：打印、串口、LSL 等刺激器抽象。

- `gui/`：主界面和弹窗。
  - `main_window.py`：系统主入口。
  - `eeg_display.py`：EEG/EMG/ECG 多区域实时波形显示。
  - `patient_dialogs.py`、`score_dialog.py`：患者和评分弹窗。

- `competition_algorithms/`：比赛分类/量化代码的封装入口。

运行时数据库已移出源码目录，默认位于项目根目录 `.runtime/swallow_experiment.db`。
