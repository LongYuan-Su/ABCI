from metabci.brainflow.abci_capture import start_data_collection  # 导入数据采集函数
import os
import serial
from datetime import datetime
from psychopy import monitors, visual, core, event
import threading
  # 导入数据采集函数


# 配置串口参数
def setup_serial_port():
    try:
        ser = serial.Serial(
            port='COM5',
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            timeout=1
        )
        print(f"串口已打开: {ser.name}")
        return ser
    except Exception as e:
        print(f"串口初始化失败: {e}")
        return None


# 创建数据文件名
def create_data_filename():
    return f"serial_data_hex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


class CustomMIExperiment:
    def __init__(self, win, ser, filename):
        self.win = win
        self.ser = ser
        self.filename = filename
        self.movies = []
        self.current_movie_index = 0

        # 获取屏幕尺寸
        self.screen_width, self.screen_height = win.size
        # 设置最大尺寸（屏幕的80%）
        self.max_width = self.screen_width * 0.8
        self.max_height = self.screen_height * 0.8

    def load_videos(self, video_paths):
        for path in video_paths:
            try:
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"找不到视频文件: {path}")

                # 创建视频对象但不设置尺寸
                movie = visual.MovieStim3(
                    self.win,
                    filename=path,
                    pos=(0, 0),  # 屏幕中心位置
                    units='pix',  # 使用像素单位
                    loop=False
                )

                # 获取视频原始尺寸
                width, height = movie.size

                # 计算等比例缩放尺寸
                width_ratio = self.max_width / width
                height_ratio = self.max_height / height
                scale_factor = min(width_ratio, height_ratio)

                # 应用等比例缩放
                movie.size = (width * scale_factor, height * scale_factor)

                self.movies.append(movie)
                print(f"加载视频: {os.path.basename(path)}, 原始尺寸: ({width}, {height}), 缩放后尺寸: {movie.size}")
            except Exception as e:
                print(f"视频加载错误: {e}")

    def play_next_video(self):
        if self.current_movie_index < len(self.movies):
            current_movie = self.movies[self.current_movie_index]
            clock = core.Clock()
            try:
                current_movie.play()
                # 启动数据采集线程 (5.5秒)
                event_label = f"Stimulus_{self.current_movie_index + 1}"
                data_thread = threading.Thread(
                    target=start_data_collection,
                    args=(self.ser, self.filename, 5.5, event_label)
                )
                data_thread.daemon = True
                data_thread.start()

                # 播放视频5.5秒
                while clock.getTime() < 5.5:
                    if current_movie.status == visual.FINISHED:
                        current_movie.seek(0)
                        current_movie.play()
                    current_movie.draw()
                    self.win.flip()
                    if event.getKeys(keyList=['q', 'escape']):
                        core.quit()
            except Exception as e:
                print(f"视频播放错误: {e}")
            finally:
                try:
                    current_movie.stop()
                    current_movie._unload()
                except:
                    pass
                self.current_movie_index += 1

    def cleanup(self):
        for movie in self.movies:
            try:
                movie.stop()
                movie._unload()
            except:
                pass


def show_start_message(win):
    message = visual.TextStim(win,
                            text="请静下心来，身体放松，评测马上开始",
                            height=40,
                            color='white',
                            pos=(-20, 0))  # 明确设置位置为屏幕中心
    message.draw()
    win.flip()
    core.wait(5.0)

def show_fixation_cross(win, ser, filename, duration=30):
    # 启动数据采集线程 (30秒)
    data_thread = threading.Thread(
        target=start_data_collection,
        args=(ser, filename, duration, "Fixation Cross")
    )
    data_thread.daemon = True
    data_thread.start()

    # 显示十字
    fixation = visual.TextStim(win, text="+", height=800, color='white')
    clock = core.Clock()
    while clock.getTime() < duration:
        fixation.draw()
        win.flip()
        if event.getKeys(keyList=['q', 'escape']):
            core.quit()


def show_start_collection_button(win):
    button = visual.TextStim(win, text="开始采集", height=40, pos=(0, 0), color='white')
    mouse = event.Mouse(win=win)
    while True:
        button.draw()
        win.flip()
        if mouse.isPressedIn(button):
            break


def show_end_screen(win):
    text = visual.TextStim(win, text="实验结束，感谢参与！", height=60, color='white')
    text.draw()
    win.flip()
    core.wait(3.0)


def run_mi_paradigm(win, ser, filename):
    video_folder = r"D:\bci\videos"
    video_files = [f"Stimulus_{i}.mp4" for i in range(1, 3)]
    video_paths = [os.path.join(video_folder, vf) for vf in video_files]

    experiment = CustomMIExperiment(win, ser, filename)
    experiment.load_videos(video_paths)

    show_start_collection_button(win)
    show_start_message(win)

    # 修改后的十字采集部分
    if ser and ser.is_open:
        show_fixation_cross(win, ser, filename, 1)  # 30秒十字采集
    else:
        # 没有串口时显示30秒十字但不采集数据
        fixation = visual.TextStim(win, text="+", height=800, color='white')
        clock = core.Clock()
        while clock.getTime() < 30:
            fixation.draw()
            win.flip()
            if event.getKeys(keyList=['q', 'escape']):
                core.quit()

    # 修改后的视频播放部分
    for _ in range(len(experiment.movies)):
        if ser and ser.is_open:
            experiment.play_next_video()  # 正常播放并采集数据
        else:
            # 没有串口时只播放视频不采集数据
            if experiment.current_movie_index < len(experiment.movies):
                current_movie = experiment.movies[experiment.current_movie_index]
                clock = core.Clock()
                try:
                    current_movie.play()
                    while clock.getTime() < 5.5:
                        if current_movie.status == visual.FINISHED:
                            current_movie.seek(0)
                            current_movie.play()
                        current_movie.draw()
                        win.flip()
                        if event.getKeys(keyList=['q', 'escape']):
                            core.quit()
                except Exception as e:
                    print(f"视频播放错误: {e}")
                finally:
                    try:
                        current_movie.stop()
                        current_movie._unload()
                    except:
                        pass
                    experiment.current_movie_index += 1

    experiment.cleanup()
    show_end_screen(win)


def main():
    # 初始化串口（允许失败）
    ser = setup_serial_port()

    # 创建数据文件（只有在串口可用时才创建）
    filename = create_data_filename() if ser and ser.is_open else None
    if filename:
        print(f"数据文件: {filename}")
    else:
        print("警告: 串口不可用，将只运行实验范式不采集数据")

    # 初始化Psychopy窗口
    mon = monitors.Monitor("primary_monitor", width=59.6, distance=60)
    mon.setSizePix([1920, 1080])
    win = visual.Window(
        size=[1920, 1080],
        fullscr=False,
        monitor=mon,
        units='pix'
    )

    try:
        # 显示开始界面
        start_text = "青少年抑郁症患者自杀意念评估系统\n\n\n按Enter键开始检测"
        start_stim = visual.TextStim(win, text=start_text, height=40, color='orange', wrapWidth=1800)
        start_stim.draw()
        win.flip()

        keys = event.waitKeys(keyList=['return', 'q', 'escape'])
        if 'q' in keys or 'escape' in keys:
            win.close()
            return

        # 运行主实验流程
        run_mi_paradigm(win, ser, filename)

    except Exception as e:
        print(f"实验运行时错误: {e}")
    finally:
        # 清理资源
        win.close()
        if ser and ser.is_open:
            ser.close()
            print("串口已关闭")


if __name__ == "__main__":
    main()