代码测试报告
一、测试项目说明
1. 项目概述
本项目是一个完整的青少年抑郁症患者自杀意念评估系统，包含：
•	EEG数据采集：通过串口实时采集脑电数据
•	视频刺激呈现：通过Psychopy播放40个标准化刺激视频
•	PLI分析：基于相位滞后指数(PLI)的脑电特征分析
•	自杀意念预测：使用线性回归模型进行自杀意念评分预测
2. 运行环境配置
组件	要求
操作系统	Windows 10/11 (64位)
Python版本	3.8+
关键依赖库	PyQt5, mne, scipy, numpy, pandas, joblib, psychopy, pyserial, scikit-learn
硬件要求	支持OpenGL的显卡，串口设备(EEG采集设备)
安装所有依赖：
pip install PyQt5 mne scipy numpy pandas joblib psychopy pyserial scikit-learn
3. 代码目录结构
metabci/
├── brainda/
│   └── algorithms/
│       └── abci_Algorithm/         # PLI分析核心代码
│           ├── abci_Algorithm.py   # 文档1：GUI分析界面
│           ├── linear_regression_model.joblib   # 预测模型
│           ├── x_scaler.joblib      # 特征标准化器
│           └── SI_PCC_pli_...csv    # 特征排序文件
│
├── brainflow/
│   └── abci_capture.py             # 文档2：EEG数据采集
│
├── brainstim/
│   └── abci_paradigm.py            # 文档5：视频刺激呈现
│
├── demo/                            # 测试目录
│   ├── main.py                      # 文档3：主入口
│   ├── demo_Algorithm.py           # 算法测试脚本
│   ├── demo_data[1-4]              # 测试数据集
│   └── videos/                      # 40个刺激视频
│       ├── Stimulus_1.mp4
│       └── ... 
________________________________________
文件格式要求：
•	CSV格式，GB2312编码
•	包含8通道EEG数据（Fp1, Fpz等）
•	包含40个Stimulus标记点
________________________________________
三、功能使用指导
1. EEG数据采集
# 进入brainflow目录
python abci_capture.py --port COM3（需要根据实际情况调整）
2. 视频刺激呈现
# 进入brainstim目录
python abci_paradigm.py
•	流程： 
1.	显示开始提示
2.	呈现注视点(3秒)
3.	播放40个视频(每个5.5秒)
4.	显示结束界面
3. PLI分析系统
# 进入demo目录
python main.py
或直接运行GUI：
python abci_Algorithm.py
GUI操作流程：
1.	点击"选择数据文件"导入EEG数据
2.	点击"开始分析"启动处理
3.	查看进度条(4个阶段)
4.	获取预测分数（0-33分）
________________________________________
四、测试要点说明
核心模块测试项
1.	数据采集模块
o	✔ 串口连接稳定性测试
o	✔ 数据包解析正确性验证
o	✔ 大文件存储压力测试
2.	刺激呈现模块
o	✔ 视频加载兼容性测试（不同分辨率/格式）
o	✔ 时间精度测试（±50ms误差）
o	✔ 异常中断恢复测试
3.	PLI分析模块
o	✔ 测试数据集验证(demo_data1-4)
o	✔ 无效数据处理（标记点≠40）
o	✔ 模型加载稳定性测试
o	✔ 计算耗时评估（<3分钟）
关键参数配置
# abci_Algorithm.py中需检查的路径
ranking_file = r"D:\...\feature_ranking.csv"  # 需更新为实际路径
model_path = 'linear_regression_model.joblib' 
scaler_path = 'x_scaler.joblib'（上述两个需要放在同一个代码上）

# abci_paradigm.py视频路径
video_folder = r"D:\MetaBCI-master\metabci\demo\videos" （需要根据实际情况调整）
五、预期输出验证
PLI分析结果示例
异常处理验证
测试场景	预期响应
缺少标记点	"标记点数量必须为40"
无效文件格式	"解析错误：CSV格式错误"
模型加载失败	"模型文件加载失败"
采集过程中断	"数据采集已安全停止"
________________________________________
六、补充说明
1.	硬件依赖：
o	需连接真实的EEG采集设备
o	建议使用专业脑电帽（8通道以上）
2.	视频资源：
o	40个刺激视频必须放置在/demo/videos/目录
o	视频命名格式：Stimulus_1.mp4 ~ Stimulus_40.mp4
3.	性能优化：
o	大文件处理启用临时目录temp
o	PLI计算使用多线程优化
4.	安全警告：
5.	# 自杀风险分级参考
6.	0-10分: 低风险
7.	10-20分: 中度关注
8.	20-30分: 高风险
>30分: 立即干预
测试人员可参考demo_data1-4的预期结果验证系统准确性，并通过修改视频路径和模型路径适配不同环境配置。
