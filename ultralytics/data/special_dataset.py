# Ultralytics YOLO 🚀, AGPL-3.0 license
import contextlib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable
from ultralytics.utils.ops import resample_segments
from .augment import Compose, Format, Instances, LetterBox, classify_augmentations, classify_transforms, v8_transforms
from .base_dataset import Base_Dataset
from .verify import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image, verify_image_label

# Ultralytics dataset *.cache_dict version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class YOLO_Dataset(Base_Dataset): #数据集加载
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        task_name (str): An explicit arg to point current task_name, Defaults to 'detect'.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data_dict=None, task_name="detect", **kwargs):
        """Initializes the YOLO_Dataset with optional configurations for segments and keypoints."""
        self.use_segments = task_name == "segment"
        self.use_keypoints = task_name == "pose"
        self.use_obb = task_name == "obb"
        self.data_dict = data_dict
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    def cache_val_save_dictcreat(self, cache_save_path=Path("./labels.cache_dict")): #验证 保存 生成字典
        """
        Cache dataset labels, check images and read shapes.

        Args:
            cache_save_path (Path): Path where to save the cache_dict file. Default is Path('./labels.cache_dict').

        Returns:
            (dict): labels.
        """
        cache_dict = {"labels": []}
        num_label_miss, num_label_found, num_label_empty, num_label_error, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {cache_save_path.parent / cache_save_path.stem}..."
        num_im_list = len(self.img_sp_list)
        nkpt, ndim = self.data_dict.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)): #关键点
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label, #函数
                iterable=zip(    #迭代
                    self.img_sp_list,
                    self.labe_sp_list,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data_dict["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = TQDM(results, desc=desc, total=num_im_list)
            # return img_sp, _label_list, img_shape, segments, keypoints, label_miss, label_found, label_empty, label_error, msg#label_miss标签文件找不到
            for img_sp, _label_list, img_shape, segments, keypoints, label_miss, label_found, label_empty, label_error, msg in pbar:
                num_label_miss += label_miss
                num_label_found += label_found
                num_label_empty += label_empty
                num_label_error += label_error
                if img_sp:
                    cache_dict["labels"].append(
                        dict(
                            img_sp=img_sp,
                            img_shape=img_shape,
                            cls=_label_list[:, 0:1],  # n, 1
                            bboxes=_label_list[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoints,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {num_label_found} images, {num_label_miss + num_label_empty} backgrounds, {num_label_error} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if num_label_found == 0:
            LOGGER.warning(f"{self.prefix}警告 ⚠️ 在以下路径没有找到标签 {cache_save_path}. {HELP_URL}")
        cache_dict["hash"] = get_hash(self.labe_sp_list + self.img_sp_list)
        cache_dict["results"] = num_label_found, num_label_miss, num_label_empty, num_label_error, len(self.img_sp_list)
        cache_dict["msgs"] = msgs  # warnings
        save_cache_file(self.prefix, cache_save_path, cache_dict)
        return cache_dict

    def get_labels_dict(self):
        """Returns dictionary of labels for YOLO training."""
        self.labe_sp_list = img2label_paths(self.img_sp_list)
        cache_path = Path(self.labe_sp_list[0]).parent.with_suffix(".cache") #标签下面的一个文件
        try:
            cache_dict, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache_dict file
            assert cache_dict["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache_dict["hash"] == get_hash(self.labe_sp_list + self.img_sp_list)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache_dict, exists = self.cache_val_save_dictcreat(cache_path), False  # run cache_dict ops

        # Display cache_dict
        num_label_found, num_label_miss, num_label_empty, num_label_error, total = cache_dict.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"扫描路径 .cache ... {num_label_found} 标签找到，{num_label_empty} 标签文件是空, {num_label_miss}标签路径找不到, {num_label_error} 错误"
            TQDM(None, desc=self.prefix + d, total=total, initial=total)  # display results
            if cache_dict["msgs"]:
                LOGGER.info("\n".join(cache_dict["msgs"]))  # display warnings
        # 文件 形状 标签类别 bbox
        # Read cache_dict
        [cache_dict.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels_dict = cache_dict["labels"] #标签字典表示一个标签文件对应的各种信息
        #                         dict(
        #                     img_sp=img_sp,
        #                     img_shape=img_shape, 
        #                     cls=_label_list[:, 0:1],  # n, 1
        #                     bboxes=_label_list[:, 1:],  # n, 4
        #                     segments=segments,
        #                     keypoints=keypoints,
        #                     normalized=True,
        #                     bbox_format="xywh",
        #                 )
        if not labels_dict:
            LOGGER.warning(f"警告 ⚠️  {cache_path}标签索引不到图像，训练可能失败. {HELP_URL}")

        self.img_sp_list = [label_dict["img_sp"] for label_dict in labels_dict]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(label_dict["cls"]), len(label_dict["bboxes"]), len(label_dict["segments"])) for label_dict in labels_dict)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths)) #长度检查
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for label_dict in labels_dict:
                label_dict["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"警告 ⚠️  {cache_path}标签索引不到分类信息，训练可能失败. {HELP_URL}")

        return labels_dict #mark 这里值生成图片路径和形状

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0 #是否进行矩形裁剪 1
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0 #0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else: #<ultralytics.data.augment.Mosaic object at 0x7f9214108e20><ultralytics.data.augment.CopyPaste object at 0x7f9204ae4bb0><ultralytics.data.augment.RandomPerspective object at 0x7f9204b06a00>
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )#<ultralytics.data.augment.Format object at 0x7f9204c0e1f0>
        return transforms
# <ultralytics.data.augment.MixUp object at 0x7f9204b0d700><ultralytics.data.augment.Albumentations object at 0x7f9204b0de80><ultralytics.data.augment.RandomHSV object at 0x7f9204b0d490><ultralytics.data.augment.RandomFlip object at 0x7f9204b0d520><ultralytics.data.augment.RandomFlip object at 0x7f9204b0daf0>
    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # list[np.array(1000, 2)] * num_samples
            # (N, 1000, 2)
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["masks", "keypoints", "bboxes", "cls", "segments", "obb"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    Extends torchvision ImageFolder to support YOLO classification tasks, offering functionalities like image
    augmentation, caching, and verification. It's designed to efficiently handle large datasets for training deep
    learning models, with optional image transformations and caching mechanisms to speed up training.

    This class allows for augmentations using both torchvision and Albumentations libraries, and supports caching images
    in RAM or on disk to reduce IO overhead during training. Additionally, it implements a robust verification process
    to ensure data integrity and consistency.

    Attributes:
        cache_ram (bool): Indicates if caching in RAM is enabled.
        cache_disk (bool): Indicates if caching on disk is enabled.
        samples (list): A list of tuples, each containing the path to an image, its class index, path to its .npy cache_dict
                        file (if caching on disk), and optionally the loaded image array (if caching in RAM).
        torch_transforms (callable): PyTorch transforms to be applied to the images.
    """

    def __init__(self, root, args, augment=False, prefix=""):
        """
        Initialize YOLO object with root, image size, augmentations, and cache_dict settings.

        Args:
            root (str): Path to the dataset directory where images are stored in a class-specific folder structure.
            args (Namespace): Configuration containing dataset-related settings such as image size, augmentation
                parameters, and cache_dict settings. It includes attributes like `imgsz` (image size), `fraction` (fraction
                of data to use), `scale`, `fliplr`, `flipud`, `cache_dict` (disk or RAM caching for faster training),
                `auto_augment`, `hsv_h`, `hsv_s`, `hsv_v`, and `crop_fraction`.
            augment (bool, optional): Whether to apply augmentations to the dataset. Default is False.
            prefix (str, optional): Prefix for logging and cache_dict filenames, aiding in dataset identification and
                debugging. Default is an empty string.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache_dict is True or args.cache_dict == "ram"  # cache_dict images into RAM
        self.cache_disk = args.cache_dict == "disk"  # cache_dict images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz, crop_fraction=args.crop_fraction)
        )

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        # Convert NumPy array to PIL image
        im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        sample = self.torch_transforms(im)
        return {"img": sample, "cls": j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f"{self.prefix}Scanning {self.root}..."
        path = Path(self.root).with_suffix(".cache_dict")  # *.cache_dict file path

        with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError):
            cache_dict = load_dataset_cache_file(path)  # attempt to load a *.cache_dict file
            assert cache_dict["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache_dict["hash"] == get_hash([x[0] for x in self.samples])  # identical hash
            num_label_found, num_label_error, total, samples = cache_dict.pop("results")  # found, missing, empty, corrupt, total
            if LOCAL_RANK in (-1, 0):
                d = f"{desc} {num_label_found} images, {num_label_error} corrupt"
                TQDM(None, desc=d, total=total, initial=total)
                if cache_dict["msgs"]:
                    LOGGER.info("\n".join(cache_dict["msgs"]))  # display warnings
            return samples

        # Run scan if *.cache_dict retrieval failed
        num_label_found, num_label_error, msgs, samples, x = 0, 0, [], [], {}
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            pbar = TQDM(results, desc=desc, total=len(self.samples))
            for sample, label_found, label_error, msg in pbar:
                if label_found:
                    samples.append(sample)
                if msg:
                    msgs.append(msg)
                num_label_found += label_found
                num_label_error += label_error
                pbar.desc = f"{desc} {num_label_found} images, {num_label_error} corrupt"
            pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        x["hash"] = get_hash([x[0] for x in self.samples])
        x["results"] = num_label_found, num_label_error, len(samples), samples
        x["msgs"] = msgs  # warnings
        save_cache_file(self.prefix, path, x)
        return samples


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache_dict dictionary from path."""
    import gc

    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache_dict = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache_dict


def save_cache_file(prefix, cache_save_path, cache_dict):
    """Save an Ultralytics dataset *.cache_dict dictionary cache_dict to cache_save_path."""
    cache_dict["version"] = DATASET_CACHE_VERSION  # add cache_dict version
    if is_dir_writeable(cache_save_path.parent):
        if cache_save_path.exists():
            cache_save_path.unlink()  # remove *.cache_dict file if exists
        np.save(str(cache_save_path), cache_dict)  # save cache_dict for next time
        cache_save_path.with_suffix(".cache.npy").rename(cache_save_path)  # remove .npy suffix
        LOGGER.info(f"{prefix}新的 cache_dict 创建路径: {cache_save_path}")
    else:
        LOGGER.warning(f"{prefix}WARNING ⚠️ 缓存目录 {cache_save_path.parent} 不可写，无法缓存.")


# TODO: support semantic segmentation
class SemanticDataset(Base_Dataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the Base_Dataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()
