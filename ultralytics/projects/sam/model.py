# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
SAM model interface.

This module provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for real-time image
segmentation tasks. The SAM model allows for promptable segmentation with unparalleled versatility in image analysis,
and has been trained on the SA-1B dataset. It features zero-shot performance capabilities, enabling it to adapt to new
image distributions and tasks without prior knowledge.

Key Features:
    - Promptable segmentation
    - Real-time performance
    - Zero-shot transfer capabilities
    - Trained on SA-1B dataset
"""

from pathlib import Path

from ultralytics.engine.engine_project import Project_Engine
from ultralytics.utils.torch_utils import model_info
from .build import build_sam
from .predict import Predictor


class SAM(Project_Engine):
    """
    SAM (Segment Anything Model) interface class.

    SAM is designed for promptable real-time image segmentation. It can be used with a variety of prompts such as
    bounding boxes, points, or labels. The model has capabilities for zero-shot performance and is trained on the SA-1B
    dataset.
    """

    def __init__(self, model="sam_b.pt") -> None:
        """
        Initializes the SAM model with a pre-trained model file.

        Args:
            model (str): Path to the pre-trained SAM model file. File should have a .pt or .pth extension.

        Raises:
            NotImplementedError: If the model file extension is not .pt or .pth.
        """
        if model and Path(model).suffix not in (".pt", ".pth"):
            raise NotImplementedError("SAM prediction requires pre-trained *.pt or *.pth model.")
        super().__init__(model=model, task_name="segment")

    def _loadProject(self, weights: str, task_name=None):
        """
        Loads the specified weights into the SAM model.

        Args:
            weights (str): Path to the weights file.
            task_name (str, optional): Task name. Defaults to None.
        """
        self.model = build_sam(weights)

    def predict(self, source, stream=False, bboxes=None, points=None, labels=None, **kwargs):
        """
        Performs segmentation prediction on the given image or video source.

        Args:
            source (str): Path to the image or video file, or a PIL.Image object, or a numpy.ndarray object.
            stream (bool, optional): If True, enables real-time streaming. Defaults to False.
            bboxes (list, optional): List of bounding box coordinates for prompted segmentation. Defaults to None.
            points (list, optional): List of points for prompted segmentation. Defaults to None.
            labels (list, optional): List of labels for prompted segmentation. Defaults to None.

        Returns:
            (list): The model predictions.
        """
        overrides = dict(conf=0.25, task_name="segment", mode="predict", imgsz=1024)
        kwargs.update(overrides)
        prompts = dict(bboxes=bboxes, points=points, labels=labels)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    def __call__(self, source=None, stream=False, bboxes=None, points=None, labels=None, **kwargs):
        """
        Alias for the 'predict' method.

        Args:
            source (str): Path to the image or video file, or a PIL.Image object, or a numpy.ndarray object.
            stream (bool, optional): If True, enables real-time streaming. Defaults to False.
            bboxes (list, optional): List of bounding box coordinates for prompted segmentation. Defaults to None.
            points (list, optional): List of points for prompted segmentation. Defaults to None.
            labels (list, optional): List of labels for prompted segmentation. Defaults to None.

        Returns:
            (list): The model predictions.
        """
        return self.predict(source, stream, bboxes, points, labels, **kwargs)

    def info(self, detailed=False, verbose=True,ptflops=False):

        return model_info(self.model, detailed=detailed, verbose=verbose,ptflops=False)

    @property
    def task_map(self):
        """
        Provides a mapping from the 'segment' task_name to its corresponding 'Predictor'.

        Returns:
            (dict): A dictionary mapping the 'segment' task_name to its corresponding 'Predictor'.
        """
        return {"segment": {"predictor": Predictor}}
