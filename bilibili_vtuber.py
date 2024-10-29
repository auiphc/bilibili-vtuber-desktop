import numpy as np
import streamlink
import threading
import pyaudio
import pygame
import ctypes
import time
import av
import os
os.environ["XDG_DATA_HOME"] = "."

from rembg import remove, new_session
from collections import deque
from configparser import ConfigParser
from DisCustomSession import DisCustomSession

config = ConfigParser()
config.read('config.ini')


def get_stream_info(stream_url: str):
    # 打开视频流
    container = av.open(stream_url)
    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)

    video_width = video_stream.width
    video_height = video_stream.height
    channels = audio_stream.codec_context.channels
    sample_rate = audio_stream.codec_context.sample_rate
    
    container.close()
    return video_width, video_height, channels, sample_rate


def process_stream(stream_url: str, model: str, model_path: str = None):
    # Global flags
    running = True
    start_audio = False

    # 获取直播流参数
    video_width, video_height, channels, sample_rate = get_stream_info(stream_url)

    # 初始化去背景模型
    if model == "isnet-custom":
        session = DisCustomSession(model, model_path=model_path,
                                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    else:
        session = new_session(model, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # 设置 PyAudio 输出流
    player = pyaudio.PyAudio()
    streamer = player.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
    volume = 0.5
    print("使用上下键调整音量")

    # 初始化 Pygame 显示窗口
    pygame.init()
    pygame.display.set_caption("Bilibili Vtuber")
    screen = pygame.display.set_mode((video_width, video_height), pygame.NOFRAME)

    # 设置窗口置顶透明
    hwnd = pygame.display.get_wm_info()['window']
    ctypes.windll.user32.SetWindowPos(hwnd, ctypes.wintypes.HWND(-1), 0, 0, 0, 0, 0x0001)
    ctypes.windll.user32.SetWindowLongW(hwnd, -20, ctypes.windll.user32.GetWindowLongW(hwnd, -20) | 0x00080000)
    ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 255, 0x00000001)


    # 初始化缓冲区
    video_buffer = deque(maxlen=1024)
    audio_buffer = deque(maxlen=1024)

    def provider():
        nonlocal running

        # 网络连接中断或信号丢失可能导致 demux 无法获取新的包, 从而导致循环结束, 所以这里加一个循环
        while running:
            # 获取音视频流
            try:
                container = av.open(stream_url)
                video_stream = next((s for s in container.streams if s.type == 'video'), None)
                audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
            except Exception as e:
                print("直播结束了")
                running = False
                return

            for packet in container.demux(video_stream, audio_stream):
                for frame in packet.decode():
                    if not running:
                        return
                    # 队列溢出
                    while len(video_buffer) > 1000 or len(audio_buffer) > 1000:
                        time.sleep(0.01)
                    # 将视频音频数据加入缓冲区
                    if packet.stream.type == 'video':
                        video_buffer.append((frame.to_ndarray(format='rgb24'), frame.time))
                    elif packet.stream.type == 'audio':
                        audio_buffer.append((frame.to_ndarray(), frame.time))

    def play_video():
        nonlocal start_audio

        while running:
            # 从缓冲区取出视频数据
            if video_buffer:
                image, frame_time = video_buffer.popleft()

                # 以下一个音频为基准, 舍弃 0.5 秒之前的帧
                if audio_buffer and audio_buffer[0][1] - frame_time > 0.5:
                    continue

                # 去除背景
                image = remove(image, session=session)
                if not start_audio:
                    start_audio = True

                # pygame 图片和 numpy 数组的读取顺序好像不一样, 这里先手动做处理
                image = np.rot90(np.fliplr(image))

                # 创建一个支持透明度的 Surface
                image_surface = pygame.Surface((image.shape[0], image.shape[1]), pygame.SRCALPHA)
                pygame.surfarray.blit_array(image_surface, image[:,:,:3])

                # 设置 alpha 通道
                alpha_surface = pygame.surfarray.pixels_alpha(image_surface)
                alpha_surface[:,:] = image[:,:,3]
                del alpha_surface

                screen.fill((0, 0, 0))
                screen.blit(image_surface, (0, 0))
                pygame.display.flip()
            else:
                time.sleep(0.001)

    def play_audio():
        while running:
            # 从缓冲区取出音频数据
            if audio_buffer and start_audio:
                audio_data, _ = audio_buffer.popleft()
                if channels == 2:
                    # 将双声道数据转换为单声道数据
                    audio_data_left = audio_data[::2]
                    audio_data_right = audio_data[1::2]
                    audio_data_mono = (audio_data_left + audio_data_right) / 2
 
                    audio_data_mono = audio_data_mono.astype(np.float32) * volume
                    streamer.write(audio_data_mono.tobytes())
                else:
                    # 单声道数据直接写入流
                    audio_data *= volume
                    streamer.write(audio_data.tobytes())
            else:
                time.sleep(0.001)


    # 启动视频音频播放线程
    provider_thread = threading.Thread(target=provider, daemon=True)
    provider_thread.start()
    video_thread = threading.Thread(target=play_video, daemon=True)
    video_thread.start()
    audio_thread = threading.Thread(target=play_audio, daemon=True)
    audio_thread.start()


    # 帧速率控制
    clock = pygame.time.Clock()

    # 标志位，用于跟踪窗口拖动
    dragging = False
    window_x, window_y = 0, 0
    mouse_offset_x, mouse_offset_y = 0, 0

    while running:
        # 检查是否退出
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            # 处理窗口拖动
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 鼠标左键按下
                    # 记录鼠标初始偏移
                    cursor = ctypes.wintypes.POINT()
                    ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))
                    mouse_offset_x, mouse_offset_y = cursor.x, cursor.y
                    # 记录窗口初始位置
                    rect = ctypes.wintypes.RECT()
                    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    window_x, window_y = rect.left, rect.top
                    dragging = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # 鼠标左键释放
                    dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    # 获取鼠标位置
                    cursor = ctypes.wintypes.POINT()
                    ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor))
                    # 计算新的窗口位置
                    new_x = window_x + cursor.x - mouse_offset_x
                    new_y = window_y + cursor.y - mouse_offset_y
                    ctypes.windll.user32.SetWindowPos(hwnd, None, new_x, new_y, 0, 0, 0x0001)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: # 调整音量
                    volume = min(volume + 0.1, 1.0)
                elif event.key == pygame.K_DOWN:
                    volume = max(volume - 0.1, 0.0)

        # 控制帧率
        clock.tick(60)

    streamer.stop_stream()
    streamer.close()
    player.terminate()
    pygame.quit()
    provider_thread.join()
    audio_thread.join()
    video_thread.join()


def main():
    # b 站直播间 id
    room_id = config.get('Settings', 'bilibili_room_id')
    streams = streamlink.streams(f"https://live.bilibili.com/{room_id}")

    if not streams or not streams['best'].url:
        print("无法找到可用的流")
        return

    model = config.get('Settings', 'rembg_model')
    model_path = config.get('Settings', 'rembg_model_path', fallback=None)
    process_stream(streams['best'].url, model, model_path)


if __name__ == "__main__":
    main()
