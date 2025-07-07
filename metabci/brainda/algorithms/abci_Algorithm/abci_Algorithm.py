import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QWidget, QMessageBox,
                             QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import mne
from scipy.signal import butter, filtfilt, hilbert
import csv
import numpy as np
import pandas as pd
from joblib import load, dump
import os


class AnalysisThread(QThread):
    """后台分析线程"""
    finished = pyqtSignal(float)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, input_file):
        super().__init__()
        self.input_file = input_file

    def run(self):
        try:
            # 执行PLI分析流程
            self.progress.emit(10)

            # 临时文件路径
            temp_dir = os.path.join(os.path.dirname(self.input_file), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            test_file = os.path.join(temp_dir, "temp_feature.csv")

            # 执行PLI分析
            result = self.PLI_main_SI(self.input_file, test_file)
            result.to_csv(test_file, index=False, header=False)
            self.progress.emit(50)

            # 加载模型和预测
            ranking_file = r"D:\MetaBCI-master\metabci\brainda\algorithms\abci_Algorithm\SI_PCC_pli_zyx_title_tt_feature_ranking.csv" #需要绝对路径引用
            model_path = 'linear_regression_model.joblib'
            scaler_path = 'x_scaler.joblib'

            # 1. 读取特征排序文件
            with open(ranking_file, 'r') as f:
                reader = csv.reader(f)
                sorted_data = list(reader)

            # 确保数据是2行
            if len(sorted_data) != 2:
                sorted_data = list(map(list, zip(*sorted_data)))

            # 提取前495个特征索引
            feature_indices = [int(idx) - 1 for idx in sorted_data[1][:495]]
            self.progress.emit(70)

            # 2. 加载模型和标准化器
            loaded_model = load(model_path)
            x_scaler = load(scaler_path)
            self.progress.emit(80)

            # 3. 读取测试数据
            with open(test_file, 'r') as f:
                reader = csv.reader(f)
                feature_data = list(reader)

            if len(feature_data) != 1:
                raise ValueError(f"测试数据应包含1行，但实际读取了{len(feature_data)}行")

            # 4. 处理并预测
            feature_vector = np.array([float(x) for x in feature_data[0]], dtype=np.float32)
            selected_features = feature_vector[feature_indices]
            selected_features_norm = x_scaler.transform(selected_features.reshape(1, -1))
            Y_pred_norm = loaded_model.predict(selected_features_norm)
            self.progress.emit(100)

            # 返回预测结果
            self.finished.emit(Y_pred_norm[0, 0])

        except Exception as e:
            self.error.emit(str(e))

    # 以下是原代码中的所有分析函数
    def PLI_main_SI(self, input_file, output_file):
        print(f'开始处理文件: {input_file}')

        # 读取CSV文件
        df = pd.read_table(input_file, encoding="GB2312", header=None, engine="python")
        data1 = df[0].str.split(',', expand=True).iloc[3:, :].reset_index(drop=True)
        data = pd.DataFrame(data1.iloc[:, 1:9].T)  # B-I列 (8通道)
        markers = pd.Series(data1.iloc[:, 10])  # J列 (标记列)

        # 检查标记点数量
        topic_indices = markers.index[markers.iloc[:].str.contains('Stimulus')].tolist()
        if len(topic_indices) != 40:
            raise ValueError(f'标记点数量必须为40，当前数量: {len(topic_indices)}')

        # 创建原始对象
        ch_names = ['Fp1', 'Fpz', 'Fp2', 'AF8', 'AF7', 'AF3', 'AFZ', 'AF4']
        sfreq = 250
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
        static_raw_cropped_resampled = raw.copy()

        # 添加事件标记
        events = np.zeros((40, 3), dtype=int)
        for i, idx in enumerate(topic_indices):
            events[i, :] = [idx, 0, i + 1]  # [样本索引, 持续时长, 事件ID]
        events_rask = events.copy()
        event_dict = {f'topic{i + 1}': i + 1 for i in range(40)}

        # 信号预处理
        static_raw_cropped_resampled.crop(tmin=10)  # 移除前10秒
        static_raw_cropped_resampled = static_raw_cropped_resampled.resample(125)  # 重采样到125Hz
        static_raw_cropped_resampled.filter(4, 30)  # 带通滤波4-30Hz

        # 计算全局平均值并校正
        global_baseline = static_raw_cropped_resampled.get_data().mean(axis=1, keepdims=True)
        corrected_data = static_raw_cropped_resampled.get_data() - global_baseline
        static_raw_cropped_resampled._data = corrected_data

        # 提取静息态数据 (前20秒)
        raw_static = static_raw_cropped_resampled.copy().crop(tmin=0, tmax=20)
        static_data = raw_static.get_data()

        # 静息态分段计算特征
        segment_length = 5 * 125  # 5秒数据
        static_features = []
        for seg in range(4):
            start_idx = seg * segment_length
            end_idx = (seg + 1) * segment_length
            seg_data = static_data[:, start_idx:end_idx]

            # 计算各频段PLI
            bands = [
                self.wave_filter(seg_data, 4, 8, 125),  # Theta
                self.wave_filter(seg_data, 8, 10, 125),  # Alpha1
                self.wave_filter(seg_data, 10, 13, 125),  # Alpha2
                self.wave_filter(seg_data, 13, 20, 125),  # Beta1
                self.wave_filter(seg_data, 20, 30, 125)  # Beta2
            ]

            band_features = [self.PLI_cal(band) for band in bands]
            net_feature = self.extract_pli_features(band_features)
            static_features.append(net_feature)

        mean_static = np.mean(static_features, axis=0)

        # 处理任务态数据
        raw_data = raw.get_data()
        cropped_data = raw_data[:, events_rask[0, 0]:]
        events_rask[:, 0] = events_rask[:, 0] - events_rask[0, 0]
        events_rask[:, 0] = events_rask[:, 0] / 2

        raw_cropped = mne.io.RawArray(cropped_data, info)
        raw_cropped.resample(125)  # 重采样到125Hz
        raw_cropped.filter(4, 30)  # 带通滤波4-30Hz

        # 基线校正
        corrected_data = raw_cropped.get_data() - global_baseline
        raw_cropped._data = corrected_data

        # 提取任务态数据
        epochs = mne.Epochs(raw_cropped, events_rask, event_id=event_dict,
                            tmin=0, tmax=5, baseline=None, preload=True)
        if len(epochs) != 40:
            raise ValueError(f'任务态分段数量异常: 预期40段，实际{len(epochs)}')

        # 计算任务态特征
        title_features = []
        for i in range(40):
            epoch_data = epochs[i].get_data()[0]  # 获取单次试验数据

            # 计算各频段PLI
            bands = [
                self.wave_filter(epoch_data, 4, 8, 125),  # Theta
                self.wave_filter(epoch_data, 8, 10, 125),  # Alpha1
                self.wave_filter(epoch_data, 10, 13, 125),  # Alpha2
                self.wave_filter(epoch_data, 13, 20, 125),  # Beta1
                self.wave_filter(epoch_data, 20, 30, 125)  # Beta2
            ]

            band_features = [self.PLI_cal(band) for band in bands]
            net_feature = self.extract_pli_features(band_features)
            title_features.append(net_feature - mean_static)

        # 计算时间差分特征
        title_features = np.array(title_features)
        time_diff_features = np.diff(title_features, axis=0)

        # 合并特征向量
        final_feature = np.hstack([
            title_features.flatten(),
            time_diff_features.flatten()
        ])

        return pd.DataFrame([final_feature])

    def wave_filter(self, data, low, high, sfreq, order=4):
        nyq = 0.5 * sfreq
        low_norm = low / nyq
        high_norm = high / nyq
        b, a = butter(order, [low_norm, high_norm], btype='band')[:2]
        return filtfilt(b, a, data)

    def PLI_cal(self, data):
        n_channels = data.shape[0]
        phase_signal = np.angle(hilbert(data))
        pli = np.zeros((n_channels, n_channels))

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = np.sin(phase_signal[i] - phase_signal[j])
                pli[i, j] = np.abs(np.mean(np.sign(phase_diff)))

        # 使矩阵对称
        pli += pli.T
        return pli

    def extract_pli_features(self, band_features):
        n_bands = len(band_features)
        n_channels = band_features[0].shape[0]
        features = []

        for pli_mat in band_features:
            # 提取上三角矩阵元素 (不包括对角线)
            triu_indices = np.triu_indices(n_channels, k=1)
            features.extend(pli_mat[triu_indices])

        return np.array(features)


class PLIAnalysisApp(QMainWindow):
    """主应用程序窗口"""

    def __init__(self):
        super().__init__()
        self.initUI()
        self.input_file = ""
        self.analysis_thread = None

    def initUI(self):
        self.setWindowTitle("EEG PLI分析系统")
        self.setGeometry(100, 100, 2000, 1300)
        # 设置窗口最小尺寸
        self.setMinimumSize(2000, 1300)
        # 允许窗口最大化
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)

        # 主部件和布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # 标题
        title_label = QLabel("自杀意念评估系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 50px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(title_label)

        # 文件选择区域
        file_group = QWidget()
        file_layout = QHBoxLayout(file_group)

        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("""
            QLabel {
                border: 1px solid #bdc3c7;
                border-radius: 10px;
                padding: 16px;
                background: #ecf0f1;
                min-width: 300px;
            }
        """)
        file_layout.addWidget(self.file_label, stretch=1)

        select_btn = QPushButton("选择数据文件")
        select_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 32px 64px;
                border-radius: 20px;
                font-size: 32px;
                
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        select_btn.clicked.connect(self.select_file)
        file_layout.addWidget(select_btn)
        layout.addWidget(file_group)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                width: 10px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # 分析按钮
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 20px;
                border-radius: 10px;
                font-size: 32px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:disabled {
                background-color: #2980b9;
            }
        """)
        self.analyze_btn.clicked.connect(self.run_analysis)
        layout.addWidget(self.analyze_btn)

        # 结果显示区域
        result_group = QWidget()
        result_layout = QVBoxLayout(result_group)

        result_title = QLabel("分析结果")
        result_title.setStyleSheet("font-size: 32px; font-weight: bold; color: #34495e;")
        result_title.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(result_title)

        self.result_label = QLabel("等待分析...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: bold;
                color: #e74c3c;
                margin: 40px;
            }
        """)
        result_layout.addWidget(self.result_label)

        # 添加分数说明标签
        score_info = QLabel("预测分数在0-33分之间，分数越高表示自杀意念越严重！")
        score_info.setAlignment(Qt.AlignCenter)
        score_info.setStyleSheet("""
                QLabel {
        font-size: 32px;
        font-weight: bold;
        color: #34495e;
        margin-top: 15px;
    }
        """)
        result_layout.addWidget(score_info)


        layout.addWidget(result_group)

        # 状态栏
        self.statusBar().showMessage("就绪")

        main_widget.setLayout(layout)

    def select_file(self):
        """选择输入文件"""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择EEG数据文件",
            "",
            "CSV文件 (*.csv);;所有文件 (*)",
            options=options
        )

        if file_name:
            self.input_file = file_name
            self.file_label.setText(os.path.basename(file_name))
            self.analyze_btn.setEnabled(True)
            self.statusBar().showMessage(f"已选择文件: {file_name}")

    def run_analysis(self):
        """执行分析"""
        if not self.input_file:
            QMessageBox.warning(self, "错误", "请先选择输入文件")
            return

        # 重置UI状态
        self.progress_bar.setValue(0)
        self.result_label.setText("分析中...")
        self.analyze_btn.setEnabled(False)
        self.statusBar().showMessage("分析中，请稍候...")

        # 创建并启动分析线程
        self.analysis_thread = AnalysisThread(self.input_file)
        self.analysis_thread.finished.connect(self.analysis_complete)
        self.analysis_thread.error.connect(self.analysis_error)
        self.analysis_thread.progress.connect(self.update_progress)
        self.analysis_thread.start()

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def analysis_complete(self, score):
        """分析完成处理"""
        self.result_label.setText(f"预测分数: {score:.4f}")
        self.analyze_btn.setEnabled(True)
        """self.statusBar().showMessage("分析完成")


            QMessageBox.information(
            self,
            "分析完成",
            f"EEG PLI分析已完成\n\n预测分数: {score:.4f}",
            QMessageBox.Ok
        )"""

    def analysis_error(self, error_msg):
        """分析错误处理"""
        self.result_label.setText("分析失败")
        self.analyze_btn.setEnabled(True)
        self.statusBar().showMessage("分析出错")

        # 显示错误消息
        QMessageBox.critical(
            self,
            "分析错误",
            f"分析过程中发生错误:\n\n{error_msg}",
            QMessageBox.Ok
        )

    def closeEvent(self, event):
        """关闭窗口事件处理"""
        if self.analysis_thread and self.analysis_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "分析仍在进行中，确定要退出吗?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.analysis_thread.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f6fa;
        }
    """)

    window = PLIAnalysisApp()
    window.show()
    sys.exit(app.exec_())