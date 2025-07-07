import sys
from PyQt5.QtWidgets import QApplication
from metabci.brainda.algorithms.abci_Algorithm.abci_Algorithm import PLIAnalysisApp  # 导入文档1的UI类
from metabci.brainstim.abci_paradigm import main as run_experiment  # 导入文档2的主函数


def main():
    # 创建QApplication实例
    app = QApplication(sys.argv)

    # 运行实验（文档2的代码）
    print("开始运行实验...")
    run_experiment()  # 这会阻塞直到实验完成

    # 实验完成后显示UI界面
    print("实验完成，显示分析界面...")
    window = PLIAnalysisApp()
    window.show()

    # 执行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()