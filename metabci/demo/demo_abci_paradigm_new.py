import os
from metabci.brainstim.paradigm import ABCI_Experiment


def main():
    """主函数整合实验流程"""
    # 初始化实验
    experiment = ABCI_Experiment()

    # 配置串口连接
    experiment.serial_manager.setup_serial_port(port='COM5', baudrate=115200)

    # 设置EEG设备（如果串口可用）
    if experiment.serial_manager.ser and experiment.serial_manager.ser.is_open:
        experiment.setup_eeg_device()

    # 设置实验窗口
    experiment.setup_window(fullscr=False)

    try:
        # 显示开始界面
        if not experiment.run_start_screen():
            return

        # 运行主实验流程
        video_folder = r"D:\MetaBCI-master\metabci\brainstim\videos"
        experiment.run_experiment(video_folder=video_folder, video_count=40)

    except Exception as e:
        print(f"实验运行时错误: {e}")
    finally:
        experiment.cleanup()
        print("实验已结束，资源已清理")


if __name__ == "__main__":
    main()