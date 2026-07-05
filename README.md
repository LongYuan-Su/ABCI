# 脑卒中吞咽困难难调控系统

本项目基于 MetaBCI 框架扩展，面向脑卒中吞咽困难场景，集成吞咽评估范式、实时 EEG/EMG/ECG 采集显示、实验事件标记、吞咽想象分类接口与 ESP32B 电刺激控制器联动等功能。

本次目录整理的原则是：尽量保持原版 MetaBCI 的核心结构不变，把本项目新增的应用层资源集中到 `applications/swallow_bci/`；必须嵌入 MetaBCI 框架才能运行的 GUI、范式、采集和算法接口仍保留在 `metabci/` 对应子包中。

## 目录结构

```text
MetaBCI_Integrated_Initial_Version/
├── metabci/                              # MetaBCI 核心包，保留原 brainda/brainflow/brainstim 组织
│   ├── brainda/
│   │   └── algorithms/deep_learning/     # 吞咽分类/量化模型加载接口等算法扩展
│   ├── brainflow/
│   │   ├── amplifiers.py                 # 原 MetaBCI 放大器抽象
│   │   ├── logger.py / workers.py        # 原 MetaBCI 日志与在线处理 Worker
│   │   ├── acquisition/                  # 数据源、采集记录、SQLite 数据库、数据查看工具
│   │   │   ├── sources.py                # OpenBCI WiFi 与 demo 数据源
│   │   │   └── recorder.py               # 采集数据、marker、患者元数据记录
│   │   ├── processing/                   # 评估、解码、特征提取、信号质量检测
│   │   ├── control/                      # 闭环调控、在线吞咽想象检测、电刺激触发接口
│   │   ├── gui/                          # 主程序页面与弹窗
│   │   │   ├── main_window.py            # 主程序入口：患者管理、范式评估、调控评估、实时脑电、日志
│   │   │   └── eeg_display.py            # EEG/EMG/ECG 多区域实时波形显示
│   │   ├── competition_algorithms/       # 分类推理、训练接口、温水吞咽量化接口
│   │   └── __init__.py                   # 对外保留常用类/函数导出
│   └── brainstim/
│       ├── paradigm_swallow.py           # 范式一：想象吞咽 + 含水/温水吞咽评估
│       ├── paradigm_swallow_part2.py     # 范式二：静息态 + 想象吞咽数据采集
│       └── paradigm_swallow_control.py   # 调控范式：识别吞咽想象后触发电刺激
├── applications/
│   └── swallow_bci/                      # 本项目新增应用层资源，区别于原 MetaBCI 框架代码
│       ├── demos/swallow_demos/          # 无硬件真实流程 demo，直接启动主 GUI 的 --demo 模式
│       ├── hardware/esp32b_controller/   # ESP32B 电刺激控制器 PlatformIO 固件工程
│       ├── models/swallow_classifier/    # 范式二吞咽想象分类模型，默认 final_model.pt
│       └── tools/esp32b_control_gui.py   # ESP32B 上位机页面与 UDP 控制接口
├── demos/                                # 原 MetaBCI 示例脚本，保留用于框架完整性
├── docs/source/                          # 原 MetaBCI 文档源文件
├── images/                               # 原 MetaBCI 图片资源
├── logs/                                 # 实验采集数据、事件标记、导入/导出结果
├── .runtime/                             # 运行时数据库和临时文件，如 swallow_experiment.db
├── tests/                                # 测试代码
├── environment.yml                       # Conda 环境描述
├── requirements.txt                      # Python 依赖
```

`logs/`、`.runtime/`、`.idea/`、`.vscode/`、`metabci.egg-info/` 等目录属于运行数据、编辑器配置或安装缓存，不是核心源码结构。

## 主要功能

1. 患者管理与实验流程入口  
   主界面提供患者选择、范式评估、调控评估、实时脑电和日志页面。导入外部日志时支持患者编号冲突处理，并可将姓名、性别、年龄、身高、体重等患者信息写入采集元数据。

2. 实时 EEG/EMG/ECG 显示与记录  
   `metabci/brainflow/gui/main_window.py` 支持 OpenBCI WiFi Shield 直连，按当前设备配置显示 9 路 EEG、6 路 EMG、1 路 ECG，并同步保存原始采集数据、事件标记和患者元数据。

3. 范式一：吞咽评估  
   `metabci/brainstim/paradigm_swallow.py` 用于吞咽评估实验，包含想象吞咽、含水提示、温水吞咽等阶段，并在每个关键阶段写入 marker，供后续按时间戳切片分析。

4. 范式二：静息 + 想象吞咽采集  
   `metabci/brainstim/paradigm_swallow_part2.py` 新增独立入口，流程为静息态采集和想象吞咽采集，主要用于后续吞咽想象/非吞咽想象分类算法训练与验证。采集结果会进入 `logs/` 下的范式二数据目录。

5. 比赛算法集成  
   `metabci/brainflow/competition_algorithms/` 封装分类推理、训练接口和温水吞咽量化接口。分类模型默认位于 `applications/swallow_bci/models/swallow_classifier/final_model.pt`，也兼容环境变量 `METABCI_SWALLOW_CLASSIFIER_MODEL` 指定的模型路径。

6. 电刺激控制器联动  
   `applications/swallow_bci/tools/esp32b_control_gui.py` 提供 ESP32B 控制器手动开关、连接测试、状态刷新和 UDP 命令发送。调控评估中勾选“识别到吞咽想象后自动触发 ESP32B 电刺激”后，主 GUI 会在每个想象吞咽提示后截取实时 EEG/EMG/ECG 5 秒窗口并调用吞咽想象分类模型；当分类置信度达到阈值时，自动向 ESP32B 发送电刺激触发命令。

7. ESP32B 固件工程  
   `applications/swallow_bci/hardware/esp32b_controller/` 是独立 PlatformIO 工程，包含 ESP32B 无线控制、电刺激通道开关、OLED 显示、蜂鸣器提示和电池状态采样等固件代码。

## 运行方式

推荐使用已经验证过的 Python 3.10 Conda 环境运行：

```powershell
D:\Anaconda\envs\metabci_2026_new\python.exe metabci\brainflow\gui\main_window.py
```

单独打开 ESP32B 控制器上位机：

```powershell
D:\Anaconda\envs\metabci_2026_new\python.exe applications\swallow_bci\tools\esp32b_control_gui.py
```

如果需要重新安装依赖，可优先参考：

```powershell
conda env create -f environment.yml
pip install -r requirements.txt
```

实际比赛环境中建议保持已经验证过的 `metabci_2026_new` 环境，避免 NumPy、SciPy、Pandas、PsychoPy 等二进制依赖被混装。

## 数据说明

实验数据默认写入 `logs/`。常见文件包括：

- 原始采集数据：保存 EEG/EMG/ECG 通道数据，单位按采集程序导出为微伏级数据。
- 事件标记文件：包含 `timestamp_sec`、`code`、`name`，用于定位实验开始、静息、想象吞咽、含水、温水吞咽等阶段。
- 范式二采集数据：用于后续吞咽想象分类模型训练或在线推理。
- 元数据文件：记录患者信息、通道信息、采样率、实验时间和保存路径等内容。

时间戳 `timestamp_sec` 是相对本次程序/实验记录起点的秒数，不是电脑系统绝对时间。分析时应将同一次实验中的采集数据和 marker 文件按相同时间轴对齐。

## 硬件连接

脑电采集使用 OpenBCI WiFi Shield 直连模式。实时脑电页面中选择 WiFi Shield 直连，填写设备 IP 和本机监听端口后连接。

电刺激控制器使用 ESP32B 无线控制。固件工程在 `applications/swallow_bci/hardware/esp32b_controller/`，上位机页面在 `applications/swallow_bci/tools/esp32b_control_gui.py`。建议先用手动开关确认硬件链路正常，再进入自动调控流程。

## 与原 MetaBCI 的关系

本项目保留 MetaBCI 原有核心目录、示例和文档：

- `metabci/brainda`：数据集与算法分析框架。
- `metabci/brainflow`：采集、记录、实时显示。
- `metabci/brainstim`：刺激呈现和实验范式。
- `demos/` 与 `docs/source/`：原框架示例和说明文档。

新增应用资源集中在 `applications/swallow_bci/`。必须与框架运行点绑定的新增代码，例如主 GUI 页面、范式脚本、数据记录器、在线分类接口，仍放在 `metabci/` 对应子包下，保证主程序入口不变。项目许可证继承原 MetaBCI 的 GPL-2.0 开源要求。

## 可运行流程 Demo

为了在没有 OpenBCI 和 ESP32B 硬件的情况下向评委展示系统流程，项目新增了 `applications/swallow_bci/demos/swallow_demos/`。这组 demo 直接启动原主程序 `metabci/brainflow/gui/main_window.py --demo`。

`--demo` 模式只替换硬件数据源：患者管理、范式评估、调控评估、实时脑电和日志页面均来自原主程序；EEG/EMG/ECG 波形仍由原来的 `MultiRegionEEGWidget` 绘制；范式一、范式二和调控评估也会启动原 `brainstim` 范式脚本。demo 使用隔离数据库 `.runtime/demo_swallow_experiment.db`，不会写入正式患者数据库。

一键入口：

```powershell
D:\Anaconda\envs\metabci_2026_new\python.exe applications\swallow_bci\demos\swallow_demos\run_all_demos.py
```

单独运行范式一演示：

```powershell
D:\Anaconda\envs\metabci_2026_new\python.exe applications\swallow_bci\demos\swallow_demos\demo_02_play_paradigm1.py
```
