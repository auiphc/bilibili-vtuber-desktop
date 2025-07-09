import numpy as np
import streamlink
import threading
import pyaudio
import pygame
import ctypes
import ctypes.wintypes
import time
import av
import os
import io
import tkinter as tk

from PIL import Image
from tkinter import simpledialog
from collections import deque
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')


def get_room_id_from_dialog():
    # Create a root window (it won't be visible)
    root = tk.Tk()
    # Hide the root window
    root.withdraw()
    
    # Show input dialog and get user input
    room_id = simpledialog.askstring(
        title="B站直播间ID",
        prompt="请输入B站直播间ID:"
    )
    
    # Destroy the root window
    root.destroy()
    
    return room_id

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


def loadModel(model_path: str, lib_path: str):
    # 载入dll
    lib_path = os.path.abspath(lib_path)
    onnx_model_lib = ctypes.CDLL(lib_path)

    # OnnxModel* OnnxModel_new(const char* onnx_model_path);
    onnx_model_lib.OnnxModel_new.argtypes = [ctypes.c_char_p]
    onnx_model_lib.OnnxModel_new.restype = ctypes.POINTER(ctypes.c_void_p)

    # unsigned char* OnnxModel_predict(OnnxModel* model, const char* data, int size, int* out_size);
    onnx_model_lib.OnnxModel_predict.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    onnx_model_lib.OnnxModel_predict.restype = ctypes.POINTER(ctypes.c_ubyte)

    # void OnnxModel_delete(OnnxModel* model);
    onnx_model_lib.OnnxModel_delete.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    onnx_model_lib.OnnxModel_delete.restype = None

    # void free_malloc(void *ptr);
    onnx_model_lib.free_malloc.argtypes = [ctypes.c_void_p]
    onnx_model_lib.free_malloc.restype = None

    model_path = ctypes.c_char_p(model_path.encode('utf-8'))
    model = onnx_model_lib.OnnxModel_new(model_path)

    return model, onnx_model_lib


def process_stream(stream_url: str, model_path: str, lib_path: str):
    # Global flags
    running = True
    start_audio = False

    # 获取直播流参数
    video_width, video_height, channels, sample_rate = get_stream_info(stream_url)

    # 初始化去背景模型
    model, onnx_model_lib = loadModel(model_path, lib_path)
    if not model:
        print("Failed to load model")
        return

    # 设置 PyAudio 输出流
    player = pyaudio.PyAudio()
    streamer = player.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, output=True)
    volume = 0.5
    print("使用上下键调整音量")
    print("使用鼠标滚轮调整窗口大小")
    print("使用 0 键重置窗口大小")

    # 初始化 Pygame 显示窗口
    pygame.init()
    pygame.display.set_caption("Bilibili Vtuber")
    screen = pygame.display.set_mode((video_width, video_height), pygame.NOFRAME)

    # 设置窗口置顶透明
    hwnd = pygame.display.get_wm_info()['window']
    ctypes.windll.user32.SetWindowPos(hwnd, ctypes.wintypes.HWND(-1), 0, 0, 0, 0, 0x0001)
    ctypes.windll.user32.SetWindowLongW(hwnd, -20, ctypes.windll.user32.GetWindowLongW(hwnd, -20) | 0x00080000)
    ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 255, 0x00000001)

    # 窗口大小调整相关变量
    current_width = video_width
    current_height = video_height
    scale_factor = 1.0
    min_scale = 0.2
    max_scale = 2.0

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
        nonlocal start_audio, current_width, current_height

        while running:
            # 从缓冲区取出视频数据
            if video_buffer:
                image, frame_time = video_buffer.popleft()

                # 以下一个音频为基准, 舍弃 0.5 秒之前的帧
                if audio_buffer and audio_buffer[0][1] - frame_time > 0.5:
                    continue

                # encode png
                image = Image.fromarray(image)
                with io.BytesIO() as output:
                    image.save(output, format="PNG")
                    data = output.getvalue()

                # 去除背景
                out_size = ctypes.c_int()
                result = onnx_model_lib.OnnxModel_predict(model, data, len(data), ctypes.byref(out_size))

                # 开始音频
                if not start_audio:
                    start_audio = True

                # decode png
                output = bytes(result[:out_size.value])
                onnx_model_lib.free_malloc(result)
                decoded_image = Image.open(io.BytesIO(output))

                # pygame 图片和 numpy 数组的读取顺序好像不一样, 这里先手动做处理
                image = np.rot90(np.fliplr(decoded_image))

                # 计算缩放比例，保持宽高比
                img_h, img_w = image.shape[:2]
                scale = min(current_width / img_w, current_height / img_h)
                scaled_width = int(img_w * scale)
                scaled_height = int(img_h * scale)

                # 缩放图像
                image = np.array(Image.fromarray(image).resize((scaled_width, scaled_height), Image.Resampling.LANCZOS))

                screen.fill((0, 0, 0))
                image_surface = pygame.surfarray.make_surface(image)
                img_rect = image_surface.get_rect(center=(current_width // 2, current_height // 2))
                screen.blit(image_surface, img_rect)
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

    # 添加窗口大小调整函数
    def resize_window(new_width, new_height):
        nonlocal current_width, current_height

        # 获取当前窗口位置和中心点
        rect = ctypes.wintypes.RECT()
        ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
        center_x = (rect.left + rect.right) // 2
        center_y = (rect.top + rect.bottom) // 2

        # 计算新位置，保持中心点不变
        new_left = center_x - new_width // 2
        new_top = center_y - new_height // 2

        # 调整窗口位置和大小
        ctypes.windll.user32.SetWindowPos(hwnd, None, new_left, new_top, new_width, new_height, 0x0004)

        # 重新设置 pygame 显示模式
        global screen
        screen = pygame.display.set_mode((new_width, new_height), pygame.NOFRAME)

        # 重新应用置顶和透明属性
        ctypes.windll.user32.SetWindowPos(hwnd, ctypes.wintypes.HWND(-1), 0, 0, 0, 0, 0x0001)
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, ctypes.windll.user32.GetWindowLongW(hwnd, -20) | 0x00080000)
        ctypes.windll.user32.SetLayeredWindowAttributes(hwnd, 0, 255, 0x00000001)

        # 更新当前尺寸记录
        current_width = new_width
        current_height = new_height

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
                elif event.key == pygame.K_0: # 重置窗口大小
                    scale_factor = 1.0
                    resize_window(video_width, video_height)
            elif event.type == pygame.MOUSEWHEEL:
                # 鼠标滚轮调整大小
                if event.y > 0: # 向上滚动，放大
                    scale_factor = min(scale_factor + 0.1, max_scale)
                else: # 向下滚动，缩小
                    scale_factor = max(scale_factor - 0.1, min_scale)
                
                new_width = int(video_width * scale_factor)
                new_height = int(video_height * scale_factor)
                resize_window(new_width, new_height)

        # 控制帧率
        clock.tick(60)

    onnx_model_lib.OnnxModel_delete(model)
    streamer.stop_stream()
    streamer.close()
    player.terminate()
    pygame.quit()
    provider_thread.join()
    audio_thread.join()
    video_thread.join()


def main():
    # b 站直播间 id
    room_id = get_room_id_from_dialog()
    if not room_id:
        print("未输入直播间ID")
        return

    streams = streamlink.streams(f"https://live.bilibili.com/{room_id}")

    if not streams or not streams['best'].url:
        print("无法找到可用的流")
        return

    model_path = config.get('Settings', 'rembg_model_path')
    lib_path = config.get('Settings', 'onnxModel_lib_path')
    process_stream(streams['best'].url, model_path, lib_path)


if __name__ == "__main__":
    main()
