# -*- mode: python ; coding: utf-8 -*-
import sys
import os


def get_pkg_path(pkg):
    return os.path.join(sys.prefix, 'Lib', 'site-packages', pkg)


a = Analysis(
    ['c_bilibili_vtuber.py'],
    pathex=[],
    binaries=[],
    datas=[
        (get_pkg_path('streamlink'), 'streamlink'),
        (get_pkg_path('attrs'), 'attrs'),
        (get_pkg_path('attr'), 'attr'),
        (get_pkg_path('trio'), 'trio'),
        (get_pkg_path('trio_websocket'), 'trio_websocket'),
        (get_pkg_path('sniffio'), 'sniffio'),
        (get_pkg_path('outcome'), 'outcome'),
        (get_pkg_path('sortedcontainers'), 'sortedcontainers'),
        (get_pkg_path('Cryptodome'), 'Cryptodome'),
        (get_pkg_path('websocket'), 'websocket'),
        (get_pkg_path('wsproto'), 'wsproto'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='c_bilibili_vtuber',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='bilibili_vtuber',
)
