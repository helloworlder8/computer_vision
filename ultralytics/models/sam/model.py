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

from ultralytics.engine.project import BaseProject
from ultralytics.utils.torch_utils import model_info

from .build import build_sam
from .predict import Predictor, SAM2Predictor



class SAM(BaseProject):
    """
    SAM (Segment Anything Model) interface class for real-time image segmentation tasks.

    This class provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for
    promptable segmentation with versatility in image analysis. It supports various prompts such as bounding
    boxes, points, or labels, and features zero-shot performance capabilities.

    Attributes:
        model (torch.nn.Module): The loaded SAM model.
        is_sam2 (bool): Indicates whether the model is SAM2 variant.
        task (str): The task type, set to "segment" for SAM models.

    Methods:
        predict: Performs segmentation prediction on the given image or video source.
        info: Logs information about the SAM model.

    Examples:
        >>> sam = SAM('sam_b.pt')
        >>> results = sam.predict('image.jpg', points=[[500, 375]])
        >>> for r in results:
        >>>     print(f"Detected {len(r.masks)} masks")
    """

    def __init__(self, model_name="sam_b.pt") -> None:
        """
        Initializes the SAM (Segment Anything Model) instance.

        Args:
            model (str): Path to the pre-trained SAM model file. File should have a .pt or .pth extension.

        Raises:
            NotImplementedError: If the model file extension is not .pt or .pth.

        Examples:
            >>> sam = SAM('sam_b.pt')
            >>> print(sam.is_sam2)
        """
        if model_name and Path(model_name).suffix not in {".pt", ".pth"}:
            raise NotImplementedError("SAM prediction requires pre-trained *.pt or *.pth model.")
        self.is_sam2 = "sam2" in Path(model_name).stem
        super().__init__(model_name=model_name, task="segment")

    def _load(self, model_pt: str, task=None):
        """
        Loads the specified weights into the SAM model.

        This method initializes the SAM model with the provided weights file, setting up the model architecture
        and loading the pre-trained parameters.

        Args:
            weights (str): Path to the weights file. Should be a .pt or .pth file containing the model parameters.
            task (str | None): Task name. If provided, it specifies the particular task the model is being loaded for.

        Examples:
            >>> sam = SAM('sam_b.pt')
            >>> sam._load('path/to/custom_weights.pt')
        """
        self.model = build_sam(model_pt)

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
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024)
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

    def info(self, detailed=False, verbose=True):
        """
        Logs information about the SAM model.

        Args:
            detailed (bool, optional): If True, displays detailed information about the model. Defaults to False.
            verbose (bool, optional): If True, displays information on the console. Defaults to True.

        Returns:
            (tuple): A tuple containing the model's information.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self):
        """
        Provides a mapping from the 'segment' task to its corresponding 'Predictor'.

        Returns:
            (Dict[str, Type[Predictor]]): A dictionary mapping the 'segment' task to its corresponding Predictor
                class. For SAM2 models, it maps to SAM2Predictor, otherwise to the standard Predictor.

        Examples:
            >>> sam = SAM('sam_b.pt')
            >>> task_map = sam.task_map
            >>> print(task_map)
            {'segment': <class 'ultralytics.models.sam.predict.Predictor'>}
        """
        return {"segment": {"predictor": SAM2Predictor if self.is_sam2 else Predictor}}
