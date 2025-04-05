# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class SegmentationTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationTrainer

        args = dict(model="yolov8n-seg.pt", data="coco8-seg.yaml", epochs=3)
        trainer = SegmentationTrainer(overrides=args)
        trainer.train()
        ```
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):

        if overrides is None:
            overrides = {}
        overrides["task"] = "segment" #再次强调
        super().__init__(cfg, overrides, _callbacks)#至少模型任务数据   图像尺寸 单通道

    def get_model(self, cfg=None, weights=None, verbose=True): #训练对象拿到模型
        """Return SegmentationModel initialized with specified config and weights."""

        ch = self.data_dict["ch"] if "ch" in self.data_dict else 3
        model = SegmentationModel(cfg, ch=ch, nc=self.data_dict["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def _get_loss_names_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.segment.SegmentationValidator(
            args=copy(self.args), dataloader=self.test_loader, save_dir=self.save_dir, _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        plot_images(
            batch["img"],
            batch["img_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_results(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png
