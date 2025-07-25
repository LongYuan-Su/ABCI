from metabci.brainflow.amplifiers import EEGParser, EEGDevice, DataCollector
import time

# 创建对象实例
parser = EEGParser()
device = EEGDevice(port='COM3')  # 替换为实际串口号
collector = DataCollector(device, parser)

try:
    # 采集5秒数据，带事件标签
    collector.start_collection(
        filename='eeg_data.txt',
        duration=5,
        event_label='baseline'
    )

    # 等待采集完成
    while collector.is_collecting:
        time.sleep(0.1)

    # 开始第二次采集
    collector.start_collection(
        filename='eeg_data.txt',
        duration=3,
        event_label='stimulus'
    )

finally:
    # 确保停止采集并关闭设备
    collector.stop_collection()
    device.close()