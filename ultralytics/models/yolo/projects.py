# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.project import BaseProject
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.utils import ROOT, yaml_load


class YOLOProject(BaseProject):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        """Initialize YOLO model or switch to YOLOWorld if model filename contains '-world'."""
        model_name = Path(model)
        if "-world" in model_name.stem and model_name.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            # Switch to YOLOWorld instance if model filename contains '-world'
            new_instance = YOLOWorld(model_name, verbose=verbose)
            self.__class__ = new_instance.__class__
            self.__dict__ = new_instance.__dict__
        else:
            # Default YOLO initialization
            super().__init__(model_name=model_name, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }
YOLO = YOLOProject




class YOLOWorld(BaseProject):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initializes the YOLOv8-World model with the given pre-trained model file. Supports *.pt and *.yaml formats.

        Args:
            model (str | Path): Path to the pre-trained model. Defaults to 'yolov8s-world.pt'.
        """
        super().__init__(model_name=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg_yaml/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel, #模型不一样
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer, #训练不一样
            }
        }

    def set_classes(self, name):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        self.model.generate_name_feats(name)
        # Remove background if it's given
        background = " "
        if background in name:
            name.remove(background)
        self.model.names = name

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = name
