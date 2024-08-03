# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.27"
# 到了这个文件夹默认提供给你这些函数类
from ultralytics.data.explorer.explorer import Explorer
from ultralytics.projects import RTDETR, SAM, YOLO, YOLOWorld
from ultralytics.projects.fastsam import FastSAM
from ultralytics.projects.nas import NAS
from ultralytics.utils import ASSETS, SETTINGS as settings
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOWorld",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
    "Explorer",
)
