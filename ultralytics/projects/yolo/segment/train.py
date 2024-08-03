# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.projects import yolo
from ultralytics.nn.tasks_model import SegmentationModel,creat_model_dict_add
from ultralytics.utils import DEFAULT_PARAM, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class SegmentationTrainer(yolo.detect.Detection_Trainer):
    """
    A class extending the Detection_Trainer class for training based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationTrainer

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml', epochs=3)
        trainer = SegmentationTrainer(overrides=args)
        trainer.DDP_or_normally_train()
        ```
    """

    def __init__(self, cfg=DEFAULT_PARAM, overrides=None, _callbacks=None):
        """Initialize a SegmentationTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task_name"] = "segment"
        super().__init__(cfg, overrides, _callbacks)

    def build_model(self, model_str=None, model=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        model_dict = creat_model_dict_add(model_str)
        model = SegmentationModel(model_dict, ch=self.data_dict["ch"], nc=self.data_dict["nc"], verbose=verbose and RANK == -1)
        if model:
            model.load(model)

        return model

    def get_validator(self):
        """Return an instance of SegmentationValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"
        return yolo.segment.SegmentationValidator(
            self.test_dataloader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["img_sp"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_result_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, on_plot=self.on_plot)  # save results.png
