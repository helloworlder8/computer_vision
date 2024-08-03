# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.engine_project import Project_Engine
from ultralytics.projects import yolo
from ultralytics.nn.tasks_model import ClassificationModel, Detection_Model, OBBModel, PoseModel, SegmentationModel, WorldModel
from ultralytics.utils import yaml_load, ROOT




class YOLOProject(Project_Engine): #项目 为了外部不大改不使用YOLOProject


    def __init__(self, model_str="yolov8n.pt", task_name=None, verbose=False): #yaml pt

        model_path = Path(model_str) 
        if "-world" in model_path.stem and model_path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(model_path)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            super().__init__(model_str=model_str, task_name=task_name, verbose=verbose) #生成一个网络模型

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel, #任务_构建大模型
                "trainer": yolo.classify.ClassificationTrainer, #工人
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": Detection_Model,#任务
                "trainer": yolo.detect.Detection_Trainer,#工人
                "validator": yolo.detect.Detection_Validator,#工人
                "predictor": yolo.detect.Detection_Predictor,#工人
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

class YOLOWorld(Project_Engine):
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt") -> None:
        """
        Initializes the YOLOv8-World model with the given pre-trained model file. Supports *.pt and *.yaml formats.

        Args:
            model (str | Path): Path to the pre-trained model. Defaults to 'yolov8s-world.pt'.
        """
        super().__init__(model=model, task_name="detect")

        # Assign default COCO class names
        self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.Detection_Validator,
                "predictor": yolo.detect.Detection_Predictor,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes
