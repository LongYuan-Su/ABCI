import os
from psychopy import monitors, visual, core, event


class CustomMIExperiment:
    def __init__(self, win):
        self.win = win
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
                    print(f"警告: 找不到视频文件: {path}")
                    continue

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
                except:
                    pass
                self.current_movie_index += 1

    def cleanup(self):
        for movie in self.movies:
            try:
                movie.stop()
            except:
                pass


def show_start_message(win):
    # 保持原始提示语不变
    message = visual.TextStim(win, text="请静下心来，身体放松，评测马上开始", height=40, color='white')
    message.draw()
    win.flip()
    core.wait(5.0)


def show_fixation_cross(win, duration=3):
    # 显示十字（保持原始大小）
    fixation = visual.TextStim(win, text="+", height=800, color='white')
    clock = core.Clock()
    while clock.getTime() < duration:
        fixation.draw()
        win.flip()


def show_end_screen(win):
    # 保持原始结束语不变
    text = visual.TextStim(win, text="实验结束，感谢参与！", height=60, color='white')
    text.draw()
    win.flip()
    core.wait(3.0)


def run_mi_paradigm(win):
    video_folder = r"D:\MetaBCI-master\metabci\demo\videos"
    video_files = [f"Stimulus_{i}.mp4" for i in range(1, 41)]  # 40个视频
    video_paths = [os.path.join(video_folder, vf) for vf in video_files]

    experiment = CustomMIExperiment(win)
    experiment.load_videos(video_paths)

    # 显示原始提示语
    show_start_message(win)

    # 显示十字（缩短为3秒便于演示）
    show_fixation_cross(win, duration=3)

    # 连续播放所有视频
    for _ in range(len(experiment.movies)):
        experiment.play_next_video()

    experiment.cleanup()
    show_end_screen(win)


def main():
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
        # 保持原始开始界面提示语不变
        start_text = "青少年抑郁症患者自杀意念评估系统\n\n\n按Enter键开始检测"
        start_stim = visual.TextStim(win, text=start_text, height=40, color='orange', wrapWidth=1800)
        start_stim.draw()
        win.flip()

        # 等待Enter键开始
        keys = event.waitKeys(keyList=['return', 'q', 'escape'])
        if 'q' in keys or 'escape' in keys:
            win.close()
            return

        # 运行主实验流程
        run_mi_paradigm(win)

    except Exception as e:
        print(f"运行错误: {e}")
    finally:
        win.close()


if __name__ == "__main__":
    main()