# Bilibili Vtuber Desktop
one background removal tool for extracting bilibili vtuber to your desktop

让b站的虚拟主播显示在你的电脑桌面！

![Example](./docs/example.gif)

## Install and Run
```
pip install -r requirements.txt

cp config.ini.example config.ini
// 把 bilibili_room_id 改为b站直播房间号

python bilibili_vtuber.py
```

## Finetune
```
git submodule update --init --recursive
```
I use SkyTNT/anime-segmentation for retraining, check [here](https://github.com/SkyTNT/anime-segmentation) for more details

我试着拿b站vtuber[雾氧Uo](https://space.bilibili.com/3493078891497932/)做了finetune, 20-50个training case应该就可以看到效果, [模型](https://modelscope.cn/models/newsea/isnet-bilibili-vtuber)可以在这里找到

## Contribution
Any contributing especially [dataset](https://modelscope.cn/datasets/newsea/isnet-bilibili-vtuber) is appreciated