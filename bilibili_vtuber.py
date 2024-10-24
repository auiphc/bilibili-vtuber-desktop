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


def process_stream(stream_url):
    running = True
    start_event = threading.Event()

    player = pyaudio.PyAudio()

    # 初始化去背景模型
    session = new_session("isnet-anime", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # 初始化缓冲区
    video_buffer = deque(maxlen=1024)
    audio_buffer = deque(maxlen=1024)

    # 打开视频流
    container = av.open(stream_url)

    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)


    # 初始化 Pygame 显示窗口
    pygame.init()
    screen = pygame.display.set_mode((video_stream.width, video_stream.height), pygame.NOFRAME)

    hwnd = pygame.display.get_wm_info()['window']
    # 设置置顶
    ctypes.windll.user32.SetWindowPos(hwnd, ctypes.wintypes.HWND(-1), 0, 0, 0, 0, 0x0001)

    # 设置窗口透明
    ctypes.windll.user32.SetWindowLongW(hwnd, -20, ctypes.windll.user32.GetWindowLongW(hwnd, -20) | 0x00080000)
    ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 255, 0x00000001)


    # 设置 PyAudio 输出流
    audio_format = pyaudio.paFloat32
    channels = audio_stream.codec_context.channels
    sample_rate = audio_stream.codec_context.sample_rate

    streamer = player.open(format=audio_format, channels=1, rate=sample_rate, output=True)


    def provider():
        # 获取音视频流
        for packet in container.demux(video_stream, audio_stream):
            for frame in packet.decode():
                # 将视频音频数据加入缓冲区
                if packet.stream.type == 'video':
                    video_buffer.append(frame.to_ndarray(format='rgb24'))
                elif packet.stream.type == 'audio':
                    audio_buffer.append(frame.to_ndarray())

    def play_video():
        while running:
            # 从缓冲区取出视频数据
            if video_buffer:
                video_data = video_buffer.popleft()
                image = np.rot90(np.fliplr(video_data))

                # 去除背景
                image = remove(image, session=session)
                start_event.set()

                image_surface = pygame.surfarray.make_surface(image[:,:,:3])
                screen.blit(image_surface, (0, 0))
                pygame.display.flip()
            else:
                time.sleep(0.001)

    def play_audio():
        while running:
            # 从缓冲区取出音频数据
            if audio_buffer and start_event.is_set():
                audio_data = audio_buffer.popleft()
                if channels == 2:
                    # 将双声道数据转换为单声道数据
                    audio_data_left = audio_data[::2]
                    audio_data_right = audio_data[1::2]
                    audio_data_mono = (audio_data_left + audio_data_right) / 2
 
                    audio_data_mono_bytes = audio_data_mono.astype(np.float32).tobytes()
                    streamer.write(audio_data_mono_bytes)
                else:
                    # 单声道数据直接写入流
                    streamer.write(audio_data.tobytes())
            else:
                time.sleep(0.001)


    # 等一段时间缓存
    provider_thread = threading.Thread(target=provider, daemon=True)
    provider_thread.start()
    time.sleep(2)

    # 启动视频音频播放线程
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
                    mouse_offset_x, mouse_offset_y = event.pos  # 记录初始偏移
                    rect = ctypes.wintypes.RECT()
                    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
                    window_x, window_y = rect.left, rect.top
                    dragging = True

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # 鼠标左键释放
                    dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    mouse_x, mouse_y = event.pos
                    # 计算新的窗口位置
                    new_x = window_x + mouse_x - mouse_offset_x
                    new_y = window_y + mouse_y - mouse_offset_y
                    ctypes.windll.user32.SetWindowPos(hwnd, None, new_x, new_y, 0, 0, 0x0001)

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
    room_id = 23222837 # b 站房间 id
    streams = streamlink.streams(f"https://live.bilibili.com/{room_id}")

    if not streams or not streams['best'].url:
        print("无法找到可用的流")
        return

    process_stream(streams['best'].url)



if __name__ == "__main__":
    main()
