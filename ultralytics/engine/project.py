# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
from pathlib import Path
from typing import List, Union
import shutil
import numpy as np
import torch
import os
from ultralytics.cfg import TASK2DATA, get_args, get_save_dir
from ultralytics.engine.results import Results
from ultralytics.hub import HUB_WEB_ROOT, HUBTrainingSession
from ultralytics.nn.tasks import load_download_model_attribute_assignment, guess_model_task, nn, create_model_dict
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    callbacks,
    checks,
    emojis,
    yaml_load,
)

# class Model(nn.Module):
class BaseProject(nn.Module):


    def __init__(
        self,
        model_name: Union[str, Path] = "yolov8n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a new instance of the YOLO model class.

        This constructor sets up the model based on the provided model path or name. It handles various types of model
        sources, including local files, Ultralytics HUB models, and Triton Server models. The method initializes several
        important attributes of the model and prepares it for operations like training, prediction, or export.

        Args:
            model (Union[str, Path], optional): The path or model file to load or create. This can be a local
                file path, a model name from Ultralytics HUB, or a Triton Server model. Defaults to 'yolov8n.pt'.
            task (Any, optional): The task type associated with the YOLO model, specifying its application domain.
                Defaults to None.
            verbose (bool, optional): If True, enables verbose output during the model's initialization and subsequent
                operations. Defaults to False.

        Raises:
            FileNotFoundError: If the specified model file does not exist or is inaccessible.
            ValueError: If the model file or configuration is invalid or unsupported.
            ImportError: If required dependencies for specific model types (like HUB SDK) are not installed.
        """
        super().__init__()
        
        self._set_init_properties(task)

        # model_name包括 yaml pt
        # Load or create new YOLO model 
        # imp """ 加载或创建模型 """
        if Path(model_name).suffix in {".yaml", ".yml"}: #model_name包括 model_yaml和model_pt
            self._new(model_name, task=task, verbose=verbose)
        else:
            self._load(model_name, task=task)



    def _set_init_properties(self,task):
        self.callbacks = callbacks.get_default_callbacks() #通用参数
        self.task = task  # task type
        self.overrides = {}  # overrides for trainer object
        self.model = None  # model object

        self.predictor = None  # reuse predictor
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        # self.model_name = None
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        
        
    def __call__(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:
        """
        An alias for the predict method, enabling the model instance to be callable.

        This method simplifies the process of making predictions by allowing the model instance to be called directly
        with the required arguments for prediction.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray, optional): The source of the image for making
                predictions. Accepts various types, including file paths, URLs, PIL images, and numpy arrays.
                Defaults to None.
            stream (bool, optional): If True, treats the input source as a continuous stream for predictions.
                Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, encapsulated in the Results class.
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """Is model a Triton Server URL string, i.e. <scheme>://<netloc>/<endpoint>/<task>"""
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:
        """Check if the provided model is a HUB model."""
        return any(
            (
                model.startswith(f"{HUB_WEB_ROOT}/models/"),  # i.e. https://hub.ultralytics.com/models/MODEL_ID
                [len(x) for x in model.split("_")] == [42, 20],  # APIKEY_MODEL
                len(model) == 20 and not Path(model).exists() and all(x not in model for x in "./\\"),  # MODEL
            )
        )

    def _new(self, model_yaml: str, task=None, verbose=False) -> None:
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str | None): model task
            model (BaseModel): Customized model.
            verbose (bool): display model info on load
        """
        model_dict = create_model_dict(model_yaml)
        
        self.model_name = model_yaml
        self.task = task or guess_model_task(model_dict) #获取任务后的操作
        self.overrides.update({"model_name": model_yaml, "task": self.task})  



        # imp""" 建模型以及属性赋值 """
        model_cls = self._task_map("model")
        self.model = model_cls(model_dict, verbose=verbose and RANK == -1)  # build model
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task


    def _load(self, model_pt: str, task=None) -> None:
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            model_pt (str): model checkpoint to be loaded
            task (str | None): model task
        """
        # if model_pt.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
        #     model_pt = checks.check_file(model_pt)  # automatically download and return local filename
        # model_pt = checks.check_model_file_from_stem(model_pt)  # add suffix, i.e. yolov8n -> yolov8n.pt

        if Path(model_pt).suffix == ".pt":
            self.ckpt, self.model = load_download_model_attribute_assignment(model_pt)
            self.model_name = self.model.model_name
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args) #任务 数据配置 图像尺寸 单一类
            self.ckpt_path = model_pt
        else:
            model_pt = checks.check_file(model_pt)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = model_pt, None
            self.task = task or guess_model_task(model_pt)
            self.ckpt_path = model_pt
        self.overrides.update({"model_name": model_pt}) #添加

    def _check_is_pytorch_model(self) -> None:
        """Raises TypeError is model is not a PyTorch model."""
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt" #还没加载的情况下
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> "BaseProject":
        """
        Resets the model parameters to randomly initialized values, effectively discarding all training information.

        This method iterates through all modules in the model and resets their parameters if they have a
        'reset_parameters' method. It also ensures that all parameters have 'requires_grad' set to True, enabling them
        to be updated during training.

        Returns:
            self (ultralytics.engine.model.Model): The instance of the class with reset weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolov8n.pt") -> "BaseProject":
        """
        Loads parameters from the specified weights file into the model.

        This method supports loading weights from a file or directly from a weights object. It matches parameters by
        name and shape and transfers them to the model.

        Args:
            weights (str | Path): Path to the weights file or a weights object. Defaults to 'yolov8n.pt'.

        Returns:
            self (ultralytics.engine.model.Model): The instance of the class with loaded weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.ckpt, weights = load_download_model_attribute_assignment(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt", use_dill=True) -> None:
        """
        Saves the current model state to a file.

        This method exports the model's checkpoint (ckpt) to the specified filename.

        Args:
            filename (str | Path): The name of the file to save the model to. Defaults to 'saved_model.pt'.
            use_dill (bool): Whether to try using dill for serialization if available. Defaults to True.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename, use_dill=use_dill)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        Logs or returns model information.

        This method provides an overview or detailed information about the model, depending on the arguments passed.
        It can control the verbosity of the output.

        Args:
            detailed (bool): If True, shows detailed information about the model. Defaults to False.
            verbose (bool): If True, prints the information. If False, returns the information. Defaults to True.

        Returns:
            (list): Various types of information about the model, depending on the 'detailed' and 'verbose' parameters.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """
        Fuses Conv2d and BatchNorm2d layers in the model.

        This method optimizes the model by fusing Conv2d and BatchNorm2d layers, which can improve inference speed.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:
        """
        Generates image embeddings based on the provided source.

        This method is a wrapper around the 'predict()' method, focusing on generating embeddings from an image source.
        It allows customization of the embedding process through various keyword arguments.

        Args:
            source (str | int | PIL.Image | np.ndarray): The source of the image for generating embeddings.
                The source can be a file path, URL, PIL image, numpy array, etc. Defaults to None.
            stream (bool): If True, predictions are streamed. Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the embedding process.

        Returns:
            (List[torch.Tensor]): A list containing the image embeddings.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs,
    ) -> List[Results]:
        """
        Performs predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode. It also provides support for SAM-type models
        through 'prompts'.

        The method sets up a new predictor if not already present and updates its arguments with each call.
        It also issues a warning and uses default assets if the 'source' is not provided. The method determines if it
        is being called from the command line interface and adjusts its behavior accordingly, including setting defaults
        for confidence threshold and saving behavior.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): The source of the image for making predictions.
                Accepts various types, including file paths, URLs, PIL images, and numpy arrays. Defaults to ASSETS.
            stream (bool, optional): Treats the input source as a continuous stream for predictions. Defaults to False.
            predictor (BasePredictor, optional): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor. Defaults to None.
            **kwargs (any): Additional keyword arguments for configuring the prediction process. These arguments allow
                for further customization of the prediction behavior.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, encapsulated in the Results class.

        Raises:
            AttributeError: If the predictor is not properly set up.
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        # is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
        #     x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        # )
        # 有什么数据，做什么任务，有什么模型，输入图像尺寸，是不是一个类
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # 人任务模型数据，图单
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = predictor or self._task_map("predictor")(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_args(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        ch = args.get("ch") if args.get("ch") else 3
        return self.predictor.predict_cli(source=source) if False else self.predictor(source=source, stream=stream,ch=ch)

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs,
    ) -> List[Results]:
        """
        Conducts object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It is
        capable of handling different types of input sources such as file paths or video streams. The method supports
        customization of the tracking process through various keyword arguments. It registers trackers if they are not
        already present and optionally persists them based on the 'persist' flag.

        The method sets a default confidence threshold specifically for ByteTrack-based tracking, which requires low
        confidence predictions as input. The tracking mode is explicitly set in the keyword arguments.

        Args:
            source (str, optional): The input source for object tracking. It can be a file path, URL, or video stream.
            stream (bool, optional): Treats the input source as a continuous video stream. Defaults to False.
            persist (bool, optional): Persists the trackers between different calls to this method. Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the tracking process. These arguments allow
                for further customization of the tracking behavior.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of tracking results, encapsulated in the Results class.

        Raises:
            AttributeError: If the predictor does not have registered trackers.
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs,
    ):

        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._task_map("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        # metrics_file = validator.metrics.save_dir/'metrics.txt'
        # with metrics_file.open('w') as f:
        #     for key, value in validator.metrics.results_dict.items():
        #         f.write(f'{key}: {value}\n')
        
        return validator.metrics

    def benchmark(
        self,
        **kwargs,
    ):
        """
        Benchmarks the model across various export formats to evaluate performance.

        This method assesses the model's performance in different export formats, such as ONNX, TorchScript, etc.
        It uses the 'benchmark' function from the ultralytics.utils.benchmarks module. The benchmarking is configured
        using a combination of default configuration values, model-specific arguments, method-specific defaults, and
        any additional user-provided keyword arguments.

        The method supports various arguments that allow customization of the benchmarking process, such as dataset
        choice, image size, precision modes, device selection, and verbosity. For a comprehensive list of all
        configurable options, users should refer to the 'configuration' section in the documentation.

        Args:
            **kwargs (any): Arbitrary keyword arguments to customize the benchmarking process. These are combined with
                default configurations, model-specific arguments, and method defaults.

        Returns:
            (dict): A dictionary containing the results of the benchmarking process.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose"),
        )

    def export(
        self,
        **kwargs,
    ) -> str:
        """
        Exports the model to a different format suitable for deployment.

        This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
        purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
        defaults, and any additional arguments provided. The combined arguments are used to configure export settings.

        The method supports a wide range of arguments to customize the export process. For a comprehensive list of all
        possible arguments, refer to the 'configuration' section in the documentation.

        Args:
            **kwargs (any): Arbitrary keyword arguments to customize the export process. These are combined with the
                model's overrides and method defaults.

        Returns:
            (str): The exported model filename in the specified format, or an object related to the export process.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {"imgsz": self.model.args["imgsz"], "batch": 1, "data": None, "verbose": False}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        trainer=None,
        **kwargs,
    ):
        # self._check_is_pytorch_model()
        # if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
        #     if any(kwargs):
        #         LOGGER.warning("WARNING ⚠️ using HUB training arguments, ignoring local training arguments.")
        #     kwargs = self.session.train_args  # overwrite kwargs

        # checks.check_pip_update_available()
        # pt 任务 数据yaml 图像尺寸 单类 模型名
        """Initialize the trainer with given parameters."""
        self._combine_args(kwargs)
        self._initialize_trainer(kwargs) #imp
        
        self.trainer.train() #imp
        
        self._update_model_and_cfg()
        map50_value = self.metrics.results_dict.get('metrics/mAP50(B)', 'unknown_value')    
        map50_str = f'{map50_value:.3f}' if isinstance(map50_value, (int, float)) else str(map50_value)
        # new_save_dir = os.path.join(self.save_dir, map50_str)   
        new_save_dir = self.metrics.save_dir.with_name(f"{self.metrics.save_dir.name}_{map50_str}") 
        # 如果 new_save_dir 已经存在，且可以删除，删除它
        if new_save_dir.exists():
            shutil.rmtree(new_save_dir)

        # 将文件从旧目录移动到新目录
        shutil.move(str(self.metrics.save_dir), str(new_save_dir))
        os.makedirs(str(self.metrics.save_dir), exist_ok=True)
        # 更新保存目录
        self.metrics.save_dir = new_save_dir
        return self.metrics


    def _combine_args(self, kwargs):
        """Initialize overrides from the configuration."""
        self.overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides #图单
        """Initialize custom arguments."""
        self.custom = {
            "data": kwargs.get("data") or self.overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model_name": self.overrides["model_name"],
            "task": self.task,
        } #强调重申
        """Combine arguments with the highest priority ones on the right."""
        self.args = {**self.overrides, **self.custom, **kwargs, "mode": "train"}
        """Set the resume parameter if applicable."""
        if self.args.get("resume"):
            self.args["resume_pt"] = self.ckpt_path #你这伏笔埋的

    def _initialize_trainer(self, kwargs):
        """Initialize the trainer."""
        trainer_cls = kwargs.get("trainer") or self._task_map("trainer")
        self.trainer = trainer_cls(overrides=self.args, _callbacks=self.callbacks) #生成文件
        if not self.args.get("resume"):
            if hasattr(self, 'model') and self.model:
                if hasattr(self.model, 'yaml'):
                    self.model.model_dict = self.model.yaml
            self.trainer.model = self.trainer.get_model(cfg=self.model.model_dict, weights=self.model if self.ckpt else None )
            self.model = self.trainer.model #你搞得不算数
        self.trainer.hub_session = self.session

        

    def _update_model_and_cfg(self):
        """Update the model and configuration after training."""
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            _, self.model = load_download_model_attribute_assignment(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)


    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args,
        **kwargs,
    ):
        """
        Conducts hyperparameter tuning for the model, with an option to use Ray Tune.

        This method supports two modes of hyperparameter tuning: using Ray Tune or a custom tuning method.
        When Ray Tune is enabled, it leverages the 'run_ray_tune' function from the ultralytics.utils.tuner module.
        Otherwise, it uses the internal 'Tuner' class for tuning. The method combines default, overridden, and
        custom arguments to configure the tuning process.

        Args:
            use_ray (bool): If True, uses Ray Tune for hyperparameter tuning. Defaults to False.
            iterations (int): The number of tuning iterations to perform. Defaults to 10.
            *args (list): Variable length argument list for additional arguments.
            **kwargs (any): Arbitrary keyword arguments. These are combined with the model's overrides and defaults.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> "BaseProject":
        """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides["device"] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self) -> list:
        """
        Retrieves the class names associated with the loaded model.

        This property returns the class names if they are defined in the model. It checks the class names for validity
        using the 'check_class_names' function from the ultralytics.nn.autobackend module.

        Returns:
            (list | None): The class names of the model if available, otherwise None.
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:  # export formats will not have predictor defined until predict() is called
            self.predictor = self._task_map("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """
        Retrieves the device on which the model's parameters are allocated.

        This property is used to determine whether the model's parameters are on CPU or GPU. It only applies to models
        that are instances of nn.Module.

        Returns:
            (torch.device | None): The device (CPU/GPU) of the model if it is a PyTorch model, otherwise None.
        """
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """
        Retrieves the transformations applied to the input data of the loaded model.

        This property returns the transformations if they are defined in the model.

        Returns:
            (object | None): The transform object of the model if available, otherwise None.
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """
        Adds a callback function for a specified event.

        This method allows the user to register a custom callback function that is triggered on a specific event during
        model training or inference.

        Args:
            event (str): The name of the event to attach the callback to.
            func (callable): The callback function to be registered.

        Raises:
            ValueError: If the event name is not recognized.
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """
        Clears all callback functions registered for a specified event.

        This method removes all custom and default callback functions associated with the given event.

        Args:
            event (str): The name of the event for which to clear the callbacks.

        Raises:
            ValueError: If the event name is not recognized.
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """
        Resets all callbacks to their default functions.

        This method reinstates the default callback functions for all events, removing any custom callbacks that were
        added previously.
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """Reset arguments when loading a PyTorch model."""
        include = {"data", "task", "imgsz", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _task_map(self, key: str):
        """Load model/trainer/validator/predictor."""
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

    @property
    def task_map(self) -> dict:
        """
        Map head to model, trainer, validator, and predictor classes.

        Returns:
            task_map (dict): The map of model task to mode classes.
        """
        raise NotImplementedError("Please provide task map for your model!")


    def profile(self, imgsz):
        if type(imgsz) is int:
            inputs = torch.randn((2, 3, imgsz, imgsz))
        else:
            inputs = torch.randn((2, 3, imgsz[0], imgsz[1]))
        if next(self.model.parameters()).device.type == 'cuda':
            return self.model.predict(inputs.to(torch.device('cuda')), profile=True)
        else:
            self.model.predict(inputs, profile=True)