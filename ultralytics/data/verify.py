# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import hashlib
import json
import os
import random
import subprocess
import time
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import is_tarfile

import cv2
import numpy as np
from PIL import Image, ImageOps

from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import (
    DATASETS_DIR,
    LOGGER,
    NUM_THREADS,
    ROOT,
    SETTINGS_YAML,
    TQDM,
    clean_url,
    colorstr,
    emojis,
    yaml_load,
    yaml_save,
)
from ultralytics.utils.checks import check_file, check_font, is_ascii
from ultralytics.utils.downloads import download, safe_download, unzip_file
from ultralytics.utils.ops import segments2boxes
from pathlib import Path

HELP_URL = "自己想想你到底错哪里了"
IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # image suffixes
VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"  # video suffixes
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders


def img2label_paths(img_sp):
    """Define label paths as a function of image paths, specifically tailored for a given directory structure."""
    # 循环处理每个图像路径
    return [x.replace(f"{os.sep}images", f"{os.sep}labels").rsplit(".", 1)[0] + ".txt" for x in img_sp]


def get_hash(paths):
    """Returns a single hash value of a list of paths (files or dirs)."""
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.sha256(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":  # only support JPEG images
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)  # the EXIF key for the orientation tag is 274
                if rotation in [6, 8]:  # rotation 270 or 90
                    s = s[1], s[0]
    return s


def verify_image(args):
    """Verify one image."""
    (img_sp, cls), prefix = args
    # Number (found, corrupt), message
    label_found, label_error, msg = 0, 0, ""
    try:
        image = Image.open(img_sp)
        image.verify()  # PIL verify
        shape = exif_size(image)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert image.format.lower() in IMG_FORMATS, f"invalid image format {image.format}"
        if image.format.lower() in ("jpg", "jpeg"):
            with open(img_sp, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(img_sp)).save(img_sp, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {img_sp}: corrupt JPEG restored and saved"
        label_found = 1
    except Exception as e:
        label_error = 1
        msg = f"{prefix}WARNING ⚠️ {img_sp}: ignoring corrupt image/label: {e}"
    return (img_sp, cls), label_found, label_error, msg


def verify_image_label(args):
    """Verify one image-label pair."""
    img_sp, label_sp, prefix, keypoint, num_cls, nkpt, ndim = args  #对应的单个图像和标签 num_cls:10 prefix:'\x1b[34m\x1b[1mtrain: \x1b[0m'
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    label_miss, label_found, label_empty, label_error, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        image = Image.open(img_sp) #str或者path都可以打开
        # image.verify()  # PIL verify
        img_shape = exif_size(image)  # image size
        img_shape = (img_shape[1], img_shape[0])  # hw
        assert (img_shape[0] > 9) & (img_shape[1] > 9), f"图像尺寸为 {img_shape} <10 像素"
        assert image.format.lower() in IMG_FORMATS, f"无效的图片格式 {image.format}"
        if image.format.lower() in ("jpg", "jpeg"):
            with open(img_sp, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(img_sp)).save(img_sp, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {img_sp}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(label_sp):
            label_found = 1  # label found
            with open(label_sp) as f:
                _label_list = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in _label_list) and (not keypoint):  # is segment
                    classes = np.array([x[0] for x in _label_list], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in _label_list]  # (cls, xy1...)
                    _label_list = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                _label_list = np.array(_label_list, dtype=np.float32)
            _num_label = len(_label_list)
            if _num_label:
                if keypoint:
                    assert _label_list.shape[1] == (5 + nkpt * ndim), f"labels require {(5 + nkpt * ndim)} columns each"
                    _xywh = _label_list[:, 5:].reshape(-1, ndim)[:, :2]
                else:
                    assert _label_list.shape[1] == 5, f"每个标签要求5列, 你咋是{_label_list.shape[1]}列 "
                    _xywh = _label_list[:, 1:]
                # assert _xywh.max() <= 1, f"坐标没有归一化或者超出了边界 {_xywh[_xywh > 1]}"
                assert _label_list.min() >= 0, f"负标签值 {_label_list[_label_list < 0]}"

                # All labels
                max_cls = _label_list[:, 0].max()  # max label count
                assert max_cls <= num_cls, (
                    f"标签类别 {int(max_cls)} 超出数据集的类别范围 {num_cls}. "
                )
                _, i = np.unique(_label_list, axis=0, return_index=True) #查重
                if len(i) < _num_label:  # duplicate row check
                    _label_list = _label_list[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {img_sp}: {_num_label - len(i)} duplicate labels removed"
            else:
                label_empty = 1  # label empty
                _label_list = np.zeros((0, (5 + nkpt * ndim) if keypoint else 5), dtype=np.float32)
        else:
            label_miss = 1  # label miss
            _label_list = np.zeros((0, (5 + nkpt * ndim) if keypoints else 5), dtype=np.float32)
        if keypoint:
            keypoints = _label_list[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (_num_label, nkpt, 3)
        _label_list = _label_list[:, :5]
        return img_sp, _label_list, img_shape, segments, keypoints, label_miss, label_found, label_empty, label_error, msg#label_miss标签文件找不到  label_empty标签文件找到了但是是空的
    except Exception as e:
        label_error = 1
        msg = f"{prefix}WARNING ⚠️ {img_sp}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, label_miss, label_found, label_empty, label_error, msg]


def polygon2mask(imgsz, polygons, color=1, downsample_ratio=1):
    """
    Convert a list of polygons to a binary mask of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int, optional): The color value to fill in the polygons on the mask. Defaults to 1.
        downsample_ratio (int, optional): Factor by which to downsample the mask. Defaults to 1.

    Returns:
        (np.ndarray): A binary mask of the specified image size with the polygons filled in.
    """
    mask = np.zeros(imgsz, dtype=np.uint8)
    polygons = np.asarray(polygons, dtype=np.int32)
    polygons = polygons.reshape((polygons.shape[0], -1, 2))
    cv2.fillPoly(mask, polygons, color=color)
    nh, nw = (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio)
    # Note: fillPoly first then resize is trying to keep the same loss calculation method when mask-ratio=1
    return cv2.resize(mask, (nw, nh))


def polygons2masks(imgsz, polygons, color, downsample_ratio=1):
    """
    Convert a list of polygons to a set of binary masks of the specified image size.

    Args:
        imgsz (tuple): The size of the image as (height, width).
        polygons (list[np.ndarray]): A list of polygons. Each polygon is an array with shape [N, M], where
                                     N is the number of polygons, and M is the number of points such that M % 2 = 0.
        color (int): The color value to fill in the polygons on the masks.
        downsample_ratio (int, optional): Factor by which to downsample each mask. Defaults to 1.

    Returns:
        (np.ndarray): A set of binary masks of the specified image size with the polygons filled in.
    """
    return np.array([polygon2mask(imgsz, [x.reshape(-1)], color, downsample_ratio) for x in polygons])


def polygons2masks_overlap(imgsz, segments, downsample_ratio=1):
    """Return a (640, 640) overlap mask."""
    masks = np.zeros(
        (imgsz[0] // downsample_ratio, imgsz[1] // downsample_ratio),
        dtype=np.int32 if len(segments) > 255 else np.uint8,
    )
    areas = []
    ms = []
    for si in range(len(segments)):
        mask = polygon2mask(imgsz, [segments[si].reshape(-1)], downsample_ratio=downsample_ratio, color=1)
        ms.append(mask)
        areas.append(mask.sum())
    areas = np.asarray(areas)
    index = np.argsort(-areas)
    ms = np.array(ms)[index]
    for i in range(len(segments)):
        mask = ms[i] * (i + 1)
        masks = masks + mask
        masks = np.clip(masks, a_min=0, a_max=i + 1)
    return masks, index


def find_dataset_yaml(path: Path) -> Path:
    """
    Find and return the YAML file associated with a Detect, Segment or Pose dataset.

    This function searches for a YAML file at the root level of the provided directory first, and if not found, it
    performs a recursive search. It prefers YAML files that have the same stem as the provided path. An AssertionError
    is raised if no YAML file is found or if multiple YAML files are found.

    Args:
        path (Path): The directory path to search for the YAML file.

    Returns:
        (Path): The path of the found YAML file.
    """
    files = list(path.glob("*.yaml")) or list(path.rglob("*.yaml"))  # try root level first and then recursive
    assert files, f"No YAML file found in '{path.resolve()}'"
    if len(files) > 1:
        files = [f for f in files if f.stem == path.stem]  # prefer *.yaml files that match
    assert len(files) == 1, f"Expected 1 YAML file in '{path.resolve()}', but found {len(files)}.\n{files}"
    return files[0]


def check_detect_dataset(data_str, autodownload=True):


    data_str = check_file(data_str)#找得到直接返回，找不到去搜索 

    # Download (optional) 压缩文件
    extract_dir = ""
    if zipfile.is_zipfile(data_str) or is_tarfile(data_str):
        new_dir = safe_download(data_str, dir=DATASETS_DIR, unzip=True, delete=False)
        data_str = find_dataset_yaml(DATASETS_DIR / new_dir)
        extract_dir, autodownload = data_str.parent, False

    # Read YAML
    data_dict = yaml_load(data_str, append_yaml_filename=True)  #可见数据集加载仅之处yaml文件加载

    # Checks
    for k in "train", "val":
        if k not in data_dict:
            raise SyntaxError(
                emojis(f"{data_str} '{k}:' 数据字典键丢失 ❌.\n在 data_yaml文件中没有找到'train' 和 'val' .")
            )

    if "names" not in data_dict and "nc" not in data_dict: #名字类别至少二选一
        raise SyntaxError(emojis(f"{data_str} 数据字典键丢失 ❌.\n 在 data_yaml文件中没有找到 'names' 或 'nc' ."))
    
    if "names" in data_dict and "nc" in data_dict and len(data_dict["names"]) != data_dict["nc"]: #长度不相等
        raise SyntaxError(emojis(f"{data_str} 'names' 长度 {len(data_dict['names'])} 和 'nc: {data_dict['nc']}' 长度不匹配."))
    
    if "names" not in data_dict: #使用默认列别
        data_dict["names"] = [f"class_{i}" for i in range(data_dict["nc"])]
    else: #计算长度
        data_dict["nc"] = len(data_dict["names"]) #长度

    data_dict["names"] = check_class_names(data_dict["names"])





    # Resolve paths
    path = Path(extract_dir or data_dict.get("path") or Path(data_dict.get("data_str", "")).parent)  # data_str root
    if not path.is_absolute():
        path = (DATASETS_DIR / path).resolve() #转成绝对路径
    data_dict["path"] = path  # 添加绝对路径
    for k in "train", "val", "test":
        if data_dict.get(k):  # prepend path
            if isinstance(data_dict[k], str):
                x = (path / data_dict[k]).resolve() #拼接
                if not x.exists() and data_dict[k].startswith("../"):
                    x = (path / data_dict[k][3:]).resolve() 
                data_dict[k] = str(x)#转成字符串
            else:
                data_dict[k] = [str((path / x).resolve()) for x in data_dict[k]]

    # Parse YAML 验证集下载
    val, s = (data_dict.get(x) for x in ("val", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            name = clean_url(data_str)  # data_str name with URL auth stripped
            m = f"\nDataset '{name}' images not found ⚠️, missing path '{[x for x in val if not x.exists()][0]}'"
            if s and autodownload: #验证不存在才下载
                LOGGER.warning(m)
            else:
                m += f"\nNote data_str download directory is '{DATASETS_DIR}'. You can update this in '{SETTINGS_YAML}'"
                raise FileNotFoundError(m)
            t = time.time()
            r = None  # success
            if s.startswith("http") and s.endswith(".zip"):  # URL
                safe_download(url=s, dir=DATASETS_DIR, delete=True)
            elif s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                r = os.system(s)
            else:  # python script
                exec(s, {"yaml": data_dict})
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}\n")
    check_font("Arial.ttf" if is_ascii(data_dict["names"]) else "Arial.Unicode.ttf")  # download fonts

    return data_dict  # dictionary





def check_cls_dataset(dataset, split=""):
    """
    Checks a classification dataset such as Imagenet.

    This function accepts a `dataset` name and attempts to retrieve the corresponding dataset information.
    If the dataset is not found locally, it attempts to download the dataset from the internet and save it locally.

    Args:
        dataset (str | Path): The name of the dataset.
        split (str, optional): The split of the dataset. Either 'val', 'test', or ''. Defaults to ''.

    Returns:
        (dict): A dictionary containing the following keys:
            - 'train' (Path): The directory path containing the training set of the dataset.
            - 'val' (Path): The directory path containing the validation set of the dataset.
            - 'test' (Path): The directory path containing the test set of the dataset.
            - 'nc' (int): The number of classes in the dataset.
            - 'names' (dict): A dictionary of class names in the dataset.
    """

    # Download (optional if dataset=https://file.zip is passed directly)
    if str(dataset).startswith(("http:/", "https:/")):
        dataset = safe_download(dataset, dir=DATASETS_DIR, unzip=True, delete=False)
    elif Path(dataset).suffix in (".zip", ".tar", ".gz"):
        file = check_file(dataset)
        dataset = safe_download(file, dir=DATASETS_DIR, unzip=True, delete=False)

    dataset = Path(dataset)
    data_dir = (dataset if dataset.is_dir() else (DATASETS_DIR / dataset)).resolve()
    if not data_dir.is_dir():
        LOGGER.warning(f"\nDataset not found ⚠️, missing path {data_dir}, attempting download...")
        t = time.time()
        if str(dataset) == "imagenet":
            subprocess.run(f"bash {ROOT / 'data/scripts/get_imagenet.sh'}", shell=True, check=True)
        else:
            url = f"https://github.com/ultralytics/yolov5/releases/download/v1.0/{dataset}.zip"
            download(url, dir=data_dir.parent)
        s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
        LOGGER.info(s)
    train_set = data_dir / "train"
    val_set = (
        data_dir / "val"
        if (data_dir / "val").exists()
        else data_dir / "validation"
        if (data_dir / "validation").exists()
        else None
    )  # data/test or data/val
    test_set = data_dir / "test" if (data_dir / "test").exists() else None  # data/val or data/test
    if split == "val" and not val_set:
        LOGGER.warning("WARNING ⚠️ Dataset 'split=val' not found, using 'split=test' instead.")
    elif split == "test" and not test_set:
        LOGGER.warning("WARNING ⚠️ Dataset 'split=test' not found, using 'split=val' instead.")

    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])  # number of classes
    names = [x.name for x in (data_dir / "train").iterdir() if x.is_dir()]  # class names list
    names = dict(enumerate(sorted(names)))

    # Print to console
    for k, v in {"train": train_set, "val": val_set, "test": test_set}.items():
        prefix = f'{colorstr(f"{k}:")} {v}...'
        if v is None:
            LOGGER.info(prefix)
        else:
            files = [path for path in v.rglob("*.*") if path.suffix[1:].lower() in IMG_FORMATS]
            label_found = len(files)  # number of files
            nd = len({file.parent for file in files})  # number of directories
            if label_found == 0:
                if k == "train":
                    raise FileNotFoundError(emojis(f"{dataset} '{k}:' no training images found ❌ "))
                else:
                    LOGGER.warning(f"{prefix} found {label_found} images in {nd} classes: WARNING ⚠️ no images found")
            elif nd != nc:
                LOGGER.warning(f"{prefix} found {label_found} images in {nd} classes: ERROR ❌️ requires {nc} classes, not {nd}")
            else:
                LOGGER.info(f"{prefix} found {label_found} images in {nd} classes ✅ ")

    return {"train": train_set, "val": val_set, "test": test_set, "nc": nc, "names": names}


class HUBDatasetStats:
    """
    A class for generating HUB dataset JSON and `-hub` dataset directory.

    Args:
        path (str): Path to data.yaml or data.zip (with data.yaml inside data.zip). Default is 'coco8.yaml'.
        task_name (str): Dataset task_name. Options are 'detect', 'segment', 'pose', 'classify'. Default is 'detect'.
        autodownload (bool): Attempt to download dataset if not found locally. Default is False.

    Example:
        Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets
            i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.
        ```python
        from ultralytics.data.utils import HUBDatasetStats

        stats = HUBDatasetStats('path/to/coco8.zip', task_name='detect')  # detect dataset
        stats = HUBDatasetStats('path/to/coco8-seg.zip', task_name='segment')  # segment dataset
        stats = HUBDatasetStats('path/to/coco8-pose.zip', task_name='pose')  # pose dataset
        stats = HUBDatasetStats('path/to/imagenet10.zip', task_name='classify')  # classification dataset

        stats.get_json(save=True)
        stats.process_images()
        ```
    """

    def __init__(self, path="coco8.yaml", task_name="detect", autodownload=False):
        """Initialize class."""
        path = Path(path).resolve()
        LOGGER.info(f"Starting HUB dataset checks for {path}....")

        self.task_name = task_name  # detect, segment, pose, classify
        if self.task_name == "classify":
            unzip_dir = unzip_file(path)
            data = check_cls_dataset(unzip_dir)
            data["path"] = unzip_dir
        else:  # detect, segment, pose
            _, data_dir, yaml_path = self._unzip(Path(path))
            try:
                # Load YAML with checks
                data = yaml_load(yaml_path)
                data["path"] = ""  # strip path since YAML should be in dataset root for all HUB datasets
                yaml_save(yaml_path, data)
                data = check_detect_dataset(yaml_path, autodownload)  # dict
                data["path"] = data_dir  # YAML path should be set to '' (relative) or parent (absolute)
            except Exception as e:
                raise Exception("error/HUB/dataset_stats/init") from e

        self.hub_dir = Path(f'{data["path"]}-hub')
        self.im_dir = self.hub_dir / "images"
        self.stats = {"nc": len(data["names"]), "names": list(data["names"].values())}  # statistics dictionary
        self.data = data

    @staticmethod
    def _unzip(path):
        """Unzip data.zip."""
        if not str(path).endswith(".zip"):  # path is data.yaml
            return False, None, path
        unzip_dir = unzip_file(path, path=path.parent)
        assert unzip_dir.is_dir(), (
            f"Error unzipping {path}, {unzip_dir} not found. " f"path/to/abc.zip MUST unzip to path/to/abc/"
        )
        return True, str(unzip_dir), find_dataset_yaml(unzip_dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f):
        """Saves a compressed image for HUB previews."""
        compress_one_image(f, self.im_dir / Path(f).name)  # save to dataset-hub

    def get_json(self, save=False, verbose=False):
        """Return dataset JSON for Ultralytics HUB."""

        def _round(labels):
            """Update labels to integer class and 4 decimal place floats."""
            if self.task_name == "detect":
                coordinates = labels["bboxes"]
            elif self.task_name == "segment":
                coordinates = [x.flatten() for x in labels["segments"]]
            elif self.task_name == "pose":
                n = labels["keypoints"].shape[0]
                coordinates = np.concatenate((labels["bboxes"], labels["keypoints"].reshape(n, -1)), 1)
            else:
                raise ValueError("Undefined dataset task_name.")
            zipped = zip(labels["cls"], coordinates)
            return [[int(c[0]), *(round(float(x), 4) for x in points)] for c, points in zipped]

        for split in "train", "val", "test":
            self.stats[split] = None  # predefine
            path = self.data.get(split)

            # Check split
            if path is None:  # no split
                continue
            files = [f for f in Path(path).rglob("*.*") if f.suffix[1:].lower() in IMG_FORMATS]  # image files in split
            if not files:  # no images
                continue

            # Get dataset statistics
            if self.task_name == "classify":
                from torchvision.datasets import ImageFolder

                dataset = ImageFolder(self.data[split])

                x = np.zeros(len(dataset.classes)).astype(int)
                for image in dataset.imgs:
                    x[image[1]] += 1

                self.stats[split] = {
                    "instance_stats": {"total": len(dataset), "per_class": x.tolist()},
                    "image_stats": {"total": len(dataset), "unlabelled": 0, "per_class": x.tolist()},
                    "labels": [{Path(k).name: v} for k, v in dataset.imgs],
                }
            else:
                from ultralytics.data import YOLO_Dataset

                dataset = YOLO_Dataset(img_path=self.data[split], data=self.data, task_name=self.task_name)
                x = np.array(
                    [
                        np.bincount(label["cls"].astype(int).flatten(), minlength=self.data["nc"])
                        for label in TQDM(dataset.labels, total=len(dataset), desc="Statistics")
                    ]
                )  # shape(128x80)
                self.stats[split] = {
                    "instance_stats": {"total": int(x.sum()), "per_class": x.sum(0).tolist()},
                    "image_stats": {
                        "total": len(dataset),
                        "unlabelled": int(np.all(x == 0, 1).sum()),
                        "per_class": (x > 0).sum(0).tolist(),
                    },
                    "labels": [{Path(k).name: _round(v)} for k, v in zip(dataset.im_files, dataset.labels)],
                }

        # Save, print and return
        if save:
            self.hub_dir.mkdir(parents=True, exist_ok=True)  # makes dataset-hub/
            stats_path = self.hub_dir / "stats.json"
            LOGGER.info(f"Saving {stats_path.resolve()}...")
            with open(stats_path, "w") as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            LOGGER.info(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):
        """Compress images for Ultralytics HUB."""
        from ultralytics.data import YOLO_Dataset  # ClassificationDataset

        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes dataset-hub/images/
        for split in "train", "val", "test":
            if self.data.get(split) is None:
                continue
            dataset = YOLO_Dataset(img_path=self.data[split], data=self.data)
            with ThreadPool(NUM_THREADS) as pool:
                for _ in TQDM(pool.imap(self._hub_ops, dataset.im_files), total=len(dataset), desc=f"{split} images"):
                    pass
        LOGGER.info(f"Done. All images saved to {self.im_dir}")
        return self.im_dir


def compress_one_image(f, f_new=None, max_dim=1920, quality=50):
    """
    Compresses a single image file to reduced size while preserving its aspect ratio and quality using either the Python
    Imaging Library (PIL) or OpenCV library. If the input image is smaller than the maximum dimension, it will not be
    resized.

    Args:
        f (str): The path to the input image file.
        f_new (str, optional): The path to the output image file. If not specified, the input file will be overwritten.
        max_dim (int, optional): The maximum dimension (width or height) of the output image. Default is 1920 pixels.
        quality (int, optional): The image compression quality as a percentage. Default is 50%.

    Example:
        ```python
        from pathlib import Path
        from ultralytics.data.utils import compress_one_image

        for f in Path('path/to/dataset').rglob('*.jpg'):
            compress_one_image(f)
        ```
    """

    try:  # use PIL
        image = Image.open(f)
        r = max_dim / max(image.height, image.width)  # ratio
        if r < 1.0:  # image too large
            image = image.resize((int(image.width * r), int(image.height * r)))
        image.save(f_new or f, "JPEG", quality=quality, optimize=True)  # save
    except Exception as e:  # use OpenCV
        LOGGER.info(f"WARNING ⚠️ HUB ops PIL failure {f}: {e}")
        image = cv2.imread(f)
        im_height, im_width = image.shape[:2]
        r = max_dim / max(im_height, im_width)  # ratio
        if r < 1.0:  # image too large
            image = cv2.resize(image, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), image)


def autosplit(path=DATASETS_DIR / "coco8/images", weights=(0.9, 0.1, 0.0), annotated_only=False):
    """
    Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt files.

    Args:
        path (Path, optional): Path to images directory. Defaults to DATASETS_DIR / 'coco8/images'.
        weights (list | tuple, optional): Train, validation, and test split fractions. Defaults to (0.9, 0.1, 0.0).
        annotated_only (bool, optional): If True, only images with an associated txt file are used. Defaults to False.

    Example:
        ```python
        from ultralytics.data.utils import autosplit

        autosplit()
        ```
    """

    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob("*.*") if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ["autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt"]  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    LOGGER.info(f"Autosplitting images from {path}" + ", using *.txt labeled images only" * annotated_only)
    for i, img in TQDM(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], "a") as f:
                f.write(f"./{img.relative_to(path.parent).as_posix()}" + "\n")  # add image to txt file



def test_verify_image_label():
    args = (
        "datasets/coco128/images/train2017/000000000009.jpg", 
        Path("datasets/coco128/labels/train2017/000000000009.txt"), 
        "\x1b[34m\x1b[1mtrain: \x1b[0m", 
        False, 
        80, 
        0, 
        0
    )
    result = verify_image_label(args)
    assert result is not None, "Returned result should not be None"
    assert len(result) == 10, "Returned result should have 10 elements"
    assert result[0] == "test_image.jpg", "Returned image file should match"
    assert isinstance(result[1], np.ndarray), "Returned label should be a numpy array"
    assert result[2] is not None, "Returned shape should not be None"
    assert result[3] is not None or result[4] is not None, "Returned segments or keypoints should not be None"
    assert result[5] in {0, 1}, "Returned label_miss should be 0 or 1"
    assert result[6] in {0, 1}, "Returned label_found should be 0 or 1"
    assert result[7] in {0, 1}, "Returned label_empty should be 0 or 1"
    assert result[8] in {0, 1}, "Returned nc should be 0 or 1"
    assert isinstance(result[9], str), "Returned msg should be a string"
if __name__ == "__main__":
    test_verify_image_label()
