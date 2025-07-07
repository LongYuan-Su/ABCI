import serial
from datetime import datetime
import time
import struct
import os
# 常量定义
SCALE_FACTOR = 0.022351744455307063  # 转换为微伏的系数
HEADER_BYTE = 0xA0
FOOTER_BYTE = 0xC0
SAMPLE_RATE = 250  # Hz

# 全局变量控制采集状态
is_collecting = False

def parse_eeg_packet(packet_data):
    """
    解析EEG数据包
    :param packet_data: 33字节的原始数据包
    :return: 解析后的数据字典或None(如果无效)
    """
    if len(packet_data) != 33:
        return None

    if packet_data[0] != HEADER_BYTE or packet_data[32] != FOOTER_BYTE:
        return None

    sample_num = packet_data[1]

    # 解析8个通道的EEG数据（每个通道3字节）
    eeg_data = []
    for i in range(8):
        start_idx = 2 + i * 3
        channel_bytes = packet_data[start_idx:start_idx + 3]

        # 将3字节转换为有符号整数
        value = struct.unpack('<i', bytes([channel_bytes[0], channel_bytes[1], channel_bytes[2], 0]))[0]
        value >>= 8  # 右移8位，因为我们只有3字节数据

        # 转换为微伏
        uV = value * SCALE_FACTOR
        eeg_data.append(uV)

    return {
        'sample_num': sample_num,
        'eeg_data': eeg_data,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    }


def start_data_collection(ser, filename, duration=None, event_label=None):
    """
    采集串口数据并保存到文件
    :param ser: 串口对象
    :param filename: 数据文件名（自动将.txt改为.csv）
    :param duration: 采集持续时间（秒）
    :param event_label: 事件标签
    """
    global is_collecting
    is_collecting = True
    start_time = time.time()
    buffer = bytearray()
    event_recorded = False

    # 确保文件扩展名是.csv
    if filename.endswith('.txt'):
        filename = filename[:-4] + '.csv'

    # 发送'b'开始数据传输
    ser.write(b'b')

    try:
        # 检查文件是否已存在且非空
        write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0

        with open(filename, 'a', newline='') as f:
            print(f"数据采集开始（{event_label}）...")
            if write_header:
                # 写入CSV头部（如果文件是新建的）
                header = "样本索引,EEG通道0,EEG通道1,EEG通道2,EEG通道3,EEG通道4,EEG通道5,EEG通道6,EEG通道7,时间戳,事件标记\n"
                f.write(header)
                f.flush()

            while is_collecting:
                if duration is not None and time.time() - start_time >= duration:
                    break

                if ser.in_waiting > 0:
                    raw_data = ser.read(ser.in_waiting)
                    buffer.extend(raw_data)

                    while len(buffer) >= 33:
                        header_pos = -1
                        for i in range(len(buffer) - 32):
                            if buffer[i] == HEADER_BYTE and buffer[i + 32] == FOOTER_BYTE:
                                header_pos = i
                                break

                        if header_pos == -1:
                            if len(buffer) > 32:
                                buffer = buffer[-32:]
                            break

                        packet = buffer[header_pos:header_pos + 33]
                        parsed = parse_eeg_packet(packet)

                        if parsed:
                            # 构建CSV格式的行
                            data_line = f"{parsed['sample_num']},"
                            data_line += ",".join([f"{val:.6f}" for val in parsed['eeg_data']])
                            data_line += f",{parsed['timestamp']},"

                            # 只在第一次记录时写入事件标签
                            if event_label and not event_recorded:
                                data_line += f"{event_label}\n"
                                event_recorded = True
                            else:
                                data_line += "\n"

                            f.write(data_line)
                            f.flush()

                        buffer = buffer[header_pos + 33:]

                time.sleep(0.001)

            print(f"数据采集结束（{event_label}）")

    except Exception as e:
        print(f"数据采集错误: {e}")
    finally:
        if not is_collecting:
            ser.write(b's')

def stop_data_collection(ser):
    """
    停止数据采集
    """
    global is_collecting
    is_collecting = False
    ser.write(b's')



def main():
    import argparse
    import serial.tools.list_ports

    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='EEG数据采集系统')
    parser.add_argument('--port', help='指定串口设备(如COM3或/dev/ttyUSB0)')
    parser.add_argument('--list-ports', action='store_true', help='列出可用串口并退出')
    args = parser.parse_args()

    # 列出可用串口
    if args.list_ports:
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("没有找到可用串口")
        else:
            print("可用串口:")
            for port in ports:
                print(f"  {port.device} - {port.description}")
        return

    # 获取或选择串口
    port_name = args.port
    if not port_name:
        ports = serial.tools.list_ports.comports()
        if not ports:
            print("错误: 没有找到可用串口")
            return
        elif len(ports) == 1:
            port_name = ports[0].device
            print(f"自动选择唯一可用串口: {port_name}")
        else:
            print("多个串口可用，请选择:")
            for i, port in enumerate(ports):
                print(f"{i + 1}. {port.device} - {port.description}")
            choice = input("输入选择(1-{}): ".format(len(ports)))
            try:
                port_name = ports[int(choice) - 1].device
            except (ValueError, IndexError):
                print("无效选择")
                return

    # 初始化串口
    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=115200,
            timeout=1
        )
        print(f"已连接串口: {port_name}")
    except Exception as e:
        print(f"无法打开串口 {port_name}: {e}")
        return

    # 主交互循环
    try:
        while True:
            print("\n选项:")
            print("1. 开始采集数据")
            print("2. 停止采集数据")
            print("3. 退出")
            choice = input("请选择(1-3): ")

            if choice == '1':
                filename = input("输入数据文件名(默认为eeg_data.csv): ") or "eeg_data.csv"
                duration = input("采集持续时间(秒，留空为持续采集): ")
                try:
                    duration = float(duration) if duration else None
                except ValueError:
                    print("无效的持续时间，将使用持续采集模式")
                    duration = None

                event_label = input("输入事件标签(可选，留空跳过): ") or None

                start_data_collection(ser, filename, duration, event_label)

            elif choice == '2':
                stop_data_collection(ser)

            elif choice == '3':
                stop_data_collection(ser)
                break

            else:
                print("无效选择")

    except KeyboardInterrupt:
        print("\n检测到Ctrl+C，正在停止采集...")
        stop_data_collection(ser)
    finally:
        ser.close()
        print("串口已关闭")


if __name__ == "__main__":
    main()