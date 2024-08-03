# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import torch

from ultralytics.projects.yolo.detect import Detection_Trainer
from ultralytics.nn.tasks_model import RTDETRDetectionModel,creat_model_dict_add
from ultralytics.utils import RANK, colorstr
from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(Detection_Trainer):
    """
    Trainer class for the RT-DETR model developed by Baidu for real-time object detection. Extends the Detection_Trainer
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
        trainer.DDP_or_normally_train()
        ```
    """

    def build_model(self, model_str=None, model=None, verbose=True):
        model_dict = creat_model_dict_add(model_str)
        RTDETR_detection_model = RTDETRDetectionModel(model_dict, ch=self.data_dict["ch"], nc=self.data_dict["nc"], verbose=verbose and RANK == -1)
        if model:
            RTDETR_detection_model.load(model)
        return RTDETR_detection_model

    def build_dataset(self, img_path, mode="val", batch=None):
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
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            prefix=colorstr(f"{mode}: "),
            data=self.data_dict,
        )

    def get_validator(self):
        """
        Returns a Detection_Validator suitable for RT-DETR model validation.

        Returns:
            (RTDETRValidator): Validator object for model validation.
        """
        self.loss_names = "giou_loss", "cls_loss", "l1_loss"
        return RTDETRValidator(self.test_dataloader, save_dir=self.save_dir, args=copy(self.args))

    def normalized_batch_images(self, batch):
        """
        Preprocess a batch of images. Scales and converts the images to float format.

        Args:
            batch (dict): Dictionary containing a batch of images, bboxes, and labels.

        Returns:
            (dict): Preprocessed batch.
        """
        batch = super().normalized_batch_images(batch)
        bs = len(batch["img"])
        batch_idx = batch["batch_idx"]
        gt_bbox, gt_class = [], []
        for i in range(bs):
            gt_bbox.append(batch["bboxes"][batch_idx == i].to(batch_idx.device))
            gt_class.append(batch["cls"][batch_idx == i].to(device=batch_idx.device, dtype=torch.long))
        return batch
