# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.projects import yolo
from ultralytics.nn.tasks_model import OBBModel,creat_model_dict_add
from ultralytics.utils import DEFAULT_PARAM, RANK


class OBBTrainer(yolo.detect.Detection_Trainer):
    """
    A class extending the Detection_Trainer class for training based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBTrainer

        args = dict(model='yolov8n-obb.pt', data='dota8.yaml', epochs=3)
        trainer = OBBTrainer(overrides=args)
        trainer.DDP_or_normally_train()
        ```
    """

    def __init__(self, cfg=DEFAULT_PARAM, overrides=None, _callbacks=None):
        """Initialize a OBBTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task_name"] = "obb"
        super().__init__(cfg, overrides, _callbacks)

    def build_model(self, model_str=None, model=None, verbose=True):
        """Return OBBModel initialized with specified config and weights."""
        model_dict = creat_model_dict_add(model_str)
        model = OBBModel(model_dict, ch=self.data_dict["ch"], nc=self.data_dict["nc"], verbose=verbose and RANK == -1)
        if model:
            model.load(model)

        return model #打架关系

    def get_validator(self):
        """Return an instance of OBBValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.obb.OBBValidator(self.test_dataloader, save_dir=self.save_dir, args=copy(self.args))
