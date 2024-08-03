# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.projects import yolo
from ultralytics.nn.tasks_model import PoseModel,creat_model_dict_add
from ultralytics.utils import DEFAULT_PARAM, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results


class PoseTrainer(yolo.detect.Detection_Trainer):
    """
    A class extending the Detection_Trainer class for training based on a pose model.

    Example:
        ```python
        from ultralytics.models.yolo.pose import PoseTrainer

        args = dict(model='yolov8n-pose.pt', data='coco8-pose.yaml', epochs=3)
        trainer = PoseTrainer(overrides=args)
        trainer.DDP_or_normally_train()
        ```
    """

    def __init__(self, cfg=DEFAULT_PARAM, overrides=None, _callbacks=None):
        """Initialize a PoseTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        overrides["task_name"] = "pose"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def build_model(self, model_str=None, model=None, verbose=True):
        """Get pose estimation model with specified configuration and weights."""
        model_dict = creat_model_dict_add(model_str)
        pose_model = PoseModel(model_dict, ch=self.data_dict["ch"], nc=self.data_dict["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose)
        if model:
            pose_model.load(model)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"
        return yolo.pose.PoseValidator(
            self.test_dataloader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
        images = batch["img"]
        kpts = batch["keypoints"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_result_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png
