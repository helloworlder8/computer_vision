# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(DetectionTrainer):
    """
    Trainer class for the RT-DETR model developed by Baidu for real-time object detection. Extends the DetectionTrainer
    class for YOLO to adapt to the specific features and architecture of RT-DETR. This model leverages Vision
    Transformers and has capabilities like IoU-aware query selection and adaptable inference speed.

    Notes:
        - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.

    Example:
        ```python
        from ultralytics.models.rtdetr.train import RTDETRTrainer

        args = dict(model='rtdetr-l.yaml', data='coco8.yaml', imgsz=640, epochs=3)
        trainer = RTDETRTrainer(overrides=args)
        trainer.train()
        ```
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration. Defaults to None.
            weights (str, optional): Path to pre-trained model weights. Defaults to None.
            verbose (bool): Verbose logging if True. Defaults to True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """

        ch = self.data_dict["ch"] if "ch" in self.data_dict else 3
        model = RTDETRDetectionModel(cfg,ch=ch, nc=self.data_dict["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def _build_dataset(self, img_path, mode="val", batch_size=None):
        """
        Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training. Defaults to None.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return RTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch_size,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data_dict=self.data_dict,
        )

    def _get_loss_names_validator(self):
        """
        Returns a DetectionValidator suitable for RT-DETR model validation.

        Returns:
            (RTDETRValidator): Validator object for model validation.
        """
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
            # def __init__(self, args=None, dataloader=None, save_dir=None, pbar=None, _callbacks=None):
        return RTDETRValidator(args=copy(self.args), dataloader=self.test_loader, save_dir=self.save_dir)

    def _normalize_img(self, batch):
        """
        Preprocess a batch of images. Scales and converts the images to float format.

        Args:
            batch (dict): Dictionary containing a batch of images, bboxes, and labels.

        Returns:
            (dict): Preprocessed batch.
        """
        batch = super()._normalize_img(batch)
        num_img = len(batch["img"]) 
        img_idx = batch["img_idx"]
        gt_bbox, gt_class = [], []
        for i in range(num_img):
            gt_bbox.append(batch["bboxes"][img_idx == i].to(img_idx.device))
            gt_class.append(batch["cls"][img_idx == i].to(device=img_idx.device, dtype=torch.long))
        return batch
