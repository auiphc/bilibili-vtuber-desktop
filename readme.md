# Bilibili Vtuber Desktop
one background removal tool for extracting bilibili vtuber to your desktop

让b站的虚拟主播显示在你的电脑桌面！

![Example](./docs/example.gif)

## Model
Model can be downloaded [here](https://huggingface.co/skytnt/anime-seg)

## Install and Run
```
build.bat

pip install -r requirements.txt

python bilibili_vtuber.py
```

## Finetune
```
git submodule update --init --recursive
```
I use SkyTNT/anime-segmentation for retraining, check [here](https://github.com/SkyTNT/anime-segmentation) for more details
