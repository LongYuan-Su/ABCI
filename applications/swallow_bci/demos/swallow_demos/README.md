# 吞咽项目真实流程 Demo

本目录提供与主程序一致的无硬件 demo。它不会另写一套假页面，而是直接启动 `metabci/brainflow/gui/main_window.py --demo`，让评委看到与正式系统一致的患者管理、范式评估、调控评估、实时脑电和日志页面。

`--demo` 模式会做三件事：

1. 使用主程序相同的 GUI 页面与范式脚本。
2. 自动创建隔离演示数据库 `.runtime/demo_swallow_experiment.db`，不写入正式患者数据库。
3. 将实时数据源切换为 `DemoSwallowAmplifier`，仍由原 `MultiRegionEEGWidget` 绘制 EEG/EMG/ECG 波形。

## 主程序 Demo

打开原 GUI，并自动连接模拟 EEG/EMG/ECG 数据源：

```powershell
D:\Anaconda\envs\metabci_2026_new\python.exe applications\swallow_bci\demos\swallow_demos\demo_01_main_gui_demo.py
```

## 范式一播放 Demo

打开原 GUI，连接模拟数据源，并自动启动范式一脚本。范式窗口弹出后，按空格开始播放流程。

```powershell
D:\Anaconda\envs\metabci_2026_new\python.exe applications\swallow_bci\demos\swallow_demos\demo_02_play_paradigm1.py
```

## 范式二播放 Demo

打开原 GUI，连接模拟数据源，并自动启动范式二脚本。范式窗口弹出后，按空格开始播放流程。

```powershell
D:\Anaconda\envs\metabci_2026_new\python.exe applications\swallow_bci\demos\swallow_demos\demo_03_play_paradigm2.py
```

## 调控评估 Demo

打开原 GUI，连接模拟数据源，并自动启动调控评估脚本。该模式会保留自动触发与手动触发电刺激的按钮；未连接真实 ESP32B 时，通信失败信息会显示在日志中。

```powershell
D:\Anaconda\envs\metabci_2026_new\python.exe applications\swallow_bci\demos\swallow_demos\demo_04_control_evaluation.py
```

## Demo 启动器

如果希望从一个入口选择不同 demo，可运行：

```powershell
D:\Anaconda\envs\metabci_2026_new\python.exe applications\swallow_bci\demos\swallow_demos\run_all_demos.py
```
