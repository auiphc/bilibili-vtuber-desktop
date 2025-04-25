# Bilibili Vtuber Desktop
one background removal tool for extracting bilibili vtuber to your desktop

让b站的虚拟主播显示在你的电脑桌面！

![Example](./docs/example.gif)

## Install and Run
```
pip install -r requirements.txt

python bilibili_vtuber.py
```

## C lib
```
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Finetune
```
git submodule update --init --recursive
```
I use SkyTNT/anime-segmentation for retraining, check [here](https://github.com/SkyTNT/anime-segmentation) for more details
