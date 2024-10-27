import yt_dlp
import argparse


def download_bilibili_video(video_id: str, cookie: str, convert: bool):
    video_url = f"https://www.bilibili.com/video/{video_id}"

    ydl_opts = { 'cookiefile': cookie } if cookie else {}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        print(ydl.list_formats(info_dict))

    format = input("Input format Id (separate by space):\n")

    ydl_opts |= {
        'format': '+'.join([x for x in format.split(' ') if x]),
        'merge_output_format': 'mp4',
    }
    if convert:
        ydl_opts |= {
            'postprocessor_args': [
                '-c:v', 'libx264',  # Set video codec to H.264
                '-preset', 'slow',  # Adjust encoding speed/quality
                '-crf', '23'        # Control quality (lower is higher quality)
            ],
        }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", type=str, help="bilibili video id")
    parser.add_argument("--cookie", type=str, default="", help="Netscape HTTP Cookie File")
    parser.add_argument("--convert", type=bool, default=False, help="set True to convert video codec to H.264 with ffmpeg")
    args = parser.parse_args()

    download_bilibili_video(args.video_id, args.cookie, args.convert)