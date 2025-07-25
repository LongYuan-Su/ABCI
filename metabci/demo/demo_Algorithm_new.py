from PyQt5.QtWidgets import QApplication

from metabci.brainda.algorithms.feature_analysis.time_freq_analysis import PLIAnalysisApp as pli_module
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f6fa;
        }
    """)

    window = pli_module.PLIAnalysisApp()
    window.show()
    sys.exit(app.exec_())