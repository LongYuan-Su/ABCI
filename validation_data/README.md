# 项目指标验证数据

本目录用于组委会复现实验基线性能指标，数据来自项目采集流程导出的 `npy/json/csv` 文件。

## 目录结构

```text
validation_data/
├── classification/
│   └── 009/
│       └── cupture_data_part2/
│           ├── 009_20260620_182128_data.npy
│           ├── 009_20260620_182128_labels.json
│           ├── 009_20260620_182128_meta.json
│           └── swallow_part2_log_20260620_180439.csv
└── quantification/
    ├── 009_subject_score_target.csv
    └── 009/
        ├── 009_20260620_180359_data.npy
        ├── 009_20260620_180359_labels.json
        ├── 009_20260620_180359_meta.json
        └── swallow_paradigm_log_20260620_175834.csv
```

## 对应验证代码

范式二吞咽想象二分类验证：

```powershell
python -m metabci.brainflow.competition_algorithms.classification_inference
```

范式一温水吞咽量化验证：

```powershell
python -m metabci.brainflow.competition_algorithms.warm_prior_regression
```

验证程序代码位于：

```text
metabci/brainflow/competition_algorithms/
```

训练好的范式二分类模型位于：

```text
applications/swallow_bci/models/swallow_classifier/final_model.pt
```

## 数据说明

- `*_data.npy`：采集到的多通道 EEG/EMG/ECG 原始数据，单位为 uV。
- `*_labels.json`：范式事件 marker，包含 `timestamp_sec`、`code`、`name` 等字段。
- `*_meta.json`：采样率、通道数、患者编号等采集元数据。
- `swallow_*_log_*.csv`：范式运行过程日志，用于辅助核对流程和时间戳。
- `009_subject_score_target.csv`：量化验证脚本用于拟合/对照的一分制目标分数。

本目录为开源仓库随附的基线验证数据，便于组委会直接复现项目指标。
