# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch

from ultralytics.cfg_yaml import TASK_TO_DATA_CFG, creat_args, creat_save_dir
from ultralytics.hub.utils import HUB_WEB_ROOT
from ultralytics.nn.tasks_model import load_pytorch_model, creat_model_task_name, nn, creat_model_dict_add
from ultralytics.utils import ASSETS, DEFAULT_PARAM_DICT, LOGGER, RANK, SETTINGS, callbacks, checks, emojis, yaml_load


class Project_Engine(nn.Module):

    def __init__(self,model_str: Union[str, Path] = "yolov8n.pt",task_name: str = None,verbose: bool = False,) -> None: #yolov8n.yaml
        # yaml_pt
        super().__init__()
        self.model_str = model_str = checks.check_stem(str(model_str).strip())  # add suffix, i.e. yolov8n -> yolov8n.pt
        self.callbacks = callbacks.get_default_callbacks() #defaultdict(<class 'list'>, {'on_pretrain_routine_start': [<function on_pretrain_routine_start at 0x7fb1a5cf8820>],
        self.task_name = task_name  # task_name type
        self.overrides = {}  # overrides for trainer object
        self.model = None  # model object

        self.predictor = None  # reuse predictor
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.model_str = None
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session



        # Load or create new YOLO model
        if Path(model_str).suffix in (".yaml", ".yml"):
            self._newProject(model_str, task_name=task_name, verbose=verbose)
        else:
            self._loadProject(model_str, task_name=task_name)



    def __call__(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:

        return self.predict(source, stream, **kwargs)

    @staticmethod
    def _get_hub_session(model: str):
        """Creates a session for Hub Training."""
        from ultralytics.hub.session import HUBTrainingSession

        session = HUBTrainingSession(model)
        return session if session.client.authenticated else None

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """Is model a Triton Server URL string, i.e. <scheme>://<netloc>/<endpoint>/<task_name>"""
        from urllib.parse import urlsplit

        url = urlsplit(model)  #SplitResult(scheme='', netloc='', path='yolov8n.pt', query='', fragment='')
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:
        """Check if the provided model is a HUB model.     类似github""" 
        return any(
            (
                model.startswith(f"{HUB_WEB_ROOT}/models/"),  # i.e. https://hub.ultralytics.com/models/MODEL_ID
                [len(x) for x in model.split("_")] == [42, 20],  # APIKEY_MODELID
                len(model) == 20 and not Path(model).exists() and all(x not in model for x in "./\\"),  # 长度 不存在不存在./字符  MODELID
            )
        )



    """ 构造成员属性大模型 传入大模型的参数和任务 """
    def _newProject(self, model_yaml: str, task_name=None, task=None, verbose=False) -> None:
        # 创建模型字典
        model_dict = creat_model_dict_add(model_yaml)

        # 任务 model_str
        self.model_str = model_yaml
        self.task_name = task_name or creat_model_task_name(model_dict)
        self.overrides.update({"model_str": model_yaml, "task_name": self.task_name})



        model_cls = task or self._task_map("model") #<class 'ultralytics.nn.tasks_model.Detection_Model'>
        self.model = model_cls(model_dict, verbose=verbose and RANK == -1) #mark class Detection_Model(Base_Model): #检测模型
        self.model.args = {**DEFAULT_PARAM_DICT, **self.overrides}
        self.model.task_name = self.task_name


    def _loadProject(self, model_pt: str, task_name=None) -> None:

        suffix = Path(model_pt).suffix #'.pt'
        if suffix == ".pt":
            self.model, self.ckpt = load_pytorch_model(model_pt)#'yolov8n.pt' ckpt还有要多少轮，最佳轮以及各种参数    模型参数（之前训练参数 ） 模型名称  模型任务名称
            self.model_str = self.model.model_str
            self.task_name = self.model.args["task_name"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args) #任务 数据配置 图像尺寸 单一类
        else: #疑点
            model_pt = checks.check_file(model_pt)
            self.model, self.ckpt = model_pt, None
            self.task_name = task_name or creat_model_task_name(model_pt)
            self.model_str = model_pt
        self.overrides.update({"model_str": model_pt, "task_name": self.task_name}) #一个添加 一个修改

    def _check_is_pytorch_model(self) -> None:
        """Raises TypeError is model is not a PyTorch model."""
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        pt_module = isinstance(self.model, nn.Module)#最大的爸爸
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )


    def _initialize_args(self, trainer=None, kwargs=None):
        if kwargs is None:
            kwargs = {}


        if hasattr(self.session, "model") and self.session.model.id:

            if any(kwargs):
                LOGGER.warning("WARNING ⚠️ using HUB training arguments, ignoring local training arguments.")
                kwargs = self.session.train_args  # 使用 HUB 训练参数

        # 设置数据配置和模型配置参数
        data_yaml = {"data_str": DEFAULT_PARAM_DICT["data_str"] or TASK_TO_DATA_CFG[self.task_name]} #默认的
        model_str = yaml_load(checks.check_yaml(kwargs.get("model_str", ""))) if kwargs.get("model_str") else self.overrides#有模型前面搭好的全作废
        bool_resume = {"resume": DEFAULT_PARAM_DICT["resume"]} #默认
        args = {**data_yaml, **model_str, **bool_resume, **kwargs, "mode": "train"}


        if args.get("resume"):
            args["resume"] = self.model_str

        # 初始化训练器


        return args #总参数

    def _create_trainer(self, trainer, args): #小参数
        # 创建任务对象
        trainer_cls = trainer or self._task_map("trainer") #<class 'ultralytics.projects.yolo.detect.train.Detection_Trainer'> 参数全传进回去了
        self.trainer = trainer_cls(overrides=args, _callbacks=self.callbacks)

        # 设置模型
        # 如果不是恢复训练，则手动设置模型
        if not args.get("resume"): # detection_model = Detection_Model(model_dict, ch=ch, nc=self.data_dict["nc"], verbose=verbose and RANK == -1)
            self.trainer.model = self.trainer.build_model(model_str=args["model_str"], model=self.model if self.ckpt else None)
            self.model = self.trainer.model
            # else:
            #     self.trainer.model= self.model #TUDO
            # 如果支持 Ultralytics HUB 并且会话不存在，则创建 HUB 模型
            if SETTINGS["hub"] is True and not self.session:
                try:
                    self.session = self._get_hub_session(self.model_str)
                    if self.session:
                        self.session.create_model(self.trainer.args)
                        # 检查模型是否成功创建
                        if not getattr(self.session.model, "id", None):
                            self.session = None
                except (PermissionError, ModuleNotFoundError):
                    # 忽略权限错误和模块未找到错误，这表示未安装 hub-sdk
                    pass

        # 将可选的 HUB 会话附加到训练器
        self.trainer.hub_session = self.session


        # 进行训练

    

    def train(self, trainer=None, **kwargs):
        # 初始化训练器
        args = self._initialize_args(trainer, kwargs)

        self._create_trainer(trainer, args) #小参数

        self.trainer.DDP_or_normally_train() #    def DDP_or_normally_train(self): #爸爸

        # 训练结束后，更新模型和配置
        if RANK in (-1, 0):
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = load_pytorch_model(ckpt)
            self.overrides = self.model.args
            # 获取验证器的指标（如果可用）
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP

        return self.metrics
        


    

    def reset_weights(self) -> "Project_Engine":
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

    def load(self, weights: Union[str, Path] = "yolov8n.pt") -> "Project_Engine":
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
            weights, self.ckpt = load_pytorch_model(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt", use_dill=True) -> None:

        self._check_is_pytorch_model()
        from ultralytics import __version__
        from datetime import datetime

        updates = {
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename, use_dill=use_dill)

    def info(self, detailed: bool = False, verbose: bool = True):

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
    ) -> list:
        """
        Args:
            source (str | int | PIL.Image | np.ndarray, optional): The source of the image for making predictions.
                Accepts various types, including file paths, URLs, PIL images, and numpy arrays. Defaults to ASSETS.
            stream (bool, optional): Treats the input source as a continuous stream for predictions. Defaults to False.
            predictor (Engine_Predictor, optional): An instance of a custom predictor class for making predictions.
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

        is_cli = (sys.argv[0].endswith("yolo") or sys.argv[0].endswith("ultralytics")) and any(
            x in sys.argv for x in ("predict", "track", "mode=predict", "mode=track")
        )
        print("代码注入！")
        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right len9
        prompts = args.pop("prompts", None)  # ->none   for SAM-type models

        if not self.predictor:
            self.predictor = predictor or self._task_map("predictor")(overrides=args, _callbacks=self.callbacks)#class{predictor}
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = creat_args(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = creat_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        ch = args.get("ch") if args.get("ch") else 3
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream,ch=ch)# 预测
    # 

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs,
    ) -> list:
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
        """
        Validates the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for a range of customization through various
        settings and configurations. It supports validation with a custom validator or the default validation approach.
        The method combines default configurations, method-specific defaults, and user-provided arguments to configure
        the validation process. After validation, it updates the model's metrics with the results obtained from the
        validator.

        The method supports various arguments that allow customization of the validation process. For a comprehensive
        list of all configurable options, users should refer to the 'configuration' section in the documentation.

        Args:
            validator (Engine_Validator, optional): An instance of a custom validator class for validating the model. If
                None, the method uses a default validator. Defaults to None.
            **kwargs (any): Arbitrary keyword arguments representing the validation configuration. These arguments are
                used to customize various aspects of the validation process.

        Returns:
            (dict): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        custom = {"rect": True}  # method defaults
        # 任务名 图像尺寸 单类 模型路径
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        # model_cls = task or self._task_map("model") #<class 'ultralytics.nn.tasks_model.Detection_Model'>
        # self.model = model_cls(model_dict, verbose=verbose and RANK == -1)

        validator_cls = validator or self._task_map("validator") #<class 'ultralytics.nn.tasks_model.Detection_Model'>
        validator =validator_cls(args=args, _callbacks=self.callbacks) #mark class Detection_Model(Base_Model): #检测模型
        # self.model这个模型是一开始就创建了
        validator(model=self.model) #def __call__(self, trainer=None, model=None):主要就是加载这个模型
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(
        self,
        **kwargs,
    ):
        """

        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}  # method defaults
        args = {**DEFAULT_PARAM_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose"),
        )

    def export(self,**kwargs,):

        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {"imgsz": self.model.args["imgsz"], "batch": 1, "data": None, "verbose": False}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)



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
            from .tuner_class import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> "Project_Engine":
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

        return check_class_names(self.model.names) if hasattr(self.model, "names") else None

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
        include = {"imgsz", "data_str", "task_name", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _task_map(self, key: str):#'trainer' 给了任务了给我具体的
        """Load model/trainer/validator/predictor."""
        try:
            return self.task_map[self.task_name][key] #一定是有任务的 dict
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task_name}' task_name yet.")
            ) from e

    @property
    def task_map(self) -> dict:
        raise NotImplementedError("Please provide task_name map for your model!")


    def profile(self, imgsz):
        if type(imgsz) is int:
            inputs = torch.randn((2, 3, imgsz, imgsz))
        else:
            inputs = torch.randn((2, 3, imgsz[0], imgsz[1]))
        if next(self.model.parameters()).device.type == 'cuda':
            return self.model.predict(inputs.to(torch.device('cuda')), profile=True)
        else:
            self.model.predict(inputs, profile=True)