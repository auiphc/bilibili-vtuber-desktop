import cv2
import os
import argparse

def extract_images(video_path, output_folder, start_time, interval, max_count):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    frame_interval = int(fps * interval)  # 每隔 interval 秒的帧数
    start_frame = int(fps * start_time)  # 计算开始的帧数

    # 设置视频的起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    image_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔 frame_interval 帧保存一张图片
        if (frame_count - start_frame) % frame_interval == 0:
            # 格式化图片编号，保存成0000-0900的格式
            image_filename = os.path.join(output_folder, f"{image_count:04}.jpg")
            cv2.imwrite(image_filename, frame)
            image_count += 1
            if image_count >= max_count:
                break
        
        frame_count += 1

    cap.release()
    print(f"导出完成，共导出 {image_count} 张图片。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_folder", type=str, help="Directory to save the output images.")
    parser.add_argument("--start_time", type=int, default=0, help="Start time in seconds to begin extracting images. Default is 0 seconds.")
    parser.add_argument("--interval", type=int, default=5, help="Interval in seconds between each extracted image. Default is 5 seconds.")
    parser.add_argument("--max_count", type=int, default=100, help="Maximum number of images to extract. Default is 100.")

    args = parser.parse_args()

    extract_images(args.video_path, args.output_folder, args.start_time, args.interval, args.max_count)
