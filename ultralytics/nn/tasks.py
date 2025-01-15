# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.extra_modules import *

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    WorldDetect,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    SegmentationLoss,
)
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    make_divisible,
    model_info,
    scale_img,
    time_sync,
)



    
try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family."""

    def forward(self, x, *args, **kwargs):#使用默认的前向传播 使用关键字传参

        if isinstance(x, dict):  # 训练阶段
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)


    def loss(self, batch, preds=None): # 目的是解耦，使得其他类是需要重载损失就可以

        if getattr(self, "criterion", None) is None: #想起来四度赤水河
            self.criterion = self.init_criterion()
        preds = self.forward(batch["img"]) if preds is None else preds #这里是正常输入返回输出结果
        return self.criterion(preds, batch) 
    
    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed) #self._predict_once前向传播

    def _predict_once(self, x, profile=False, visualize=False, embed=None): #torch.Size([2, 3, 640, 640])

        y, dt, embeddings = [], [], []  # outputs
        for m in self.seqential_model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # list
            if profile:
                self._profile_one_layer(m, x, dt)
            # x = m(x)  # 前向传播
            x = m(x)  # 前向传播
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to
        the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.seqential_model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.seqential_model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.seqential_model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
            self.model_info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def model_info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information.

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        if hasattr(self, 'model'):
            self.seqential_model = self.model
        m = self.seqential_model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(self.seqential_model.state_dict())} items from pretrained weights")



    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg="yolov8n.yaml", ch=3, nc=None, verbose=True):  #配置 通道 类别 详细输出
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        self.model_dict = cfg if isinstance(cfg, dict) else create_model_dict(cfg)
        self._check_deprecated_module()
        self._build_model(ch, nc, verbose)
        self._build_strides(ch)
        initialize_weights(self)
        if verbose:
            self.model_info()
            LOGGER.info("")

    # def _init_model_dict(self, cfg):
    #     """Initialize the model configuration dictionary."""
    #     self.model_dict = cfg if isinstance(cfg, dict) else create_model_dict(cfg)

    def _check_deprecated_module(self):
        """Check for deprecated modules and log warnings if necessary."""
        if self.model_dict["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "WARNING ⚠️ YOLOv9 `Silence` module is deprecated in favor of nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.model_dict["backbone"][0][2] = "nn.Identity"

    def _build_model(self, ch, nc, verbose):
        """Define the model structure."""
        # 如果模型字典中没有 "ch" 键，则设置为传入的 "ch" 值
        self.model_dict.setdefault("ch", ch)  # 使用 setdefault 简化获取和设置

        # 如果传入的 nc 与当前字典中的 nc 不同，更新并记录日志
        if nc and nc != self.model_dict.get("nc"):
            LOGGER.info(f"Overriding model.yaml nc={self.model_dict.get('nc')} with nc={nc}")
            self.model_dict["nc"] = nc

        
        self.seqential_model, self.save = parse_model(deepcopy(self.model_dict), ch=self.model_dict['ch'], verbose=verbose)
        self.names = {i: f"{i}" for i in range(self.model_dict["nc"])}
        self.inplace = self.model_dict.get("inplace", True)
        self.end2end = getattr(self.seqential_model[-1], "end2end", False)

    def _build_strides(self, ch):
        """Build the strides for the model."""
        m = self.seqential_model[-1]
        if isinstance(m, Detect):
            s = 256
            
            def _forward(x):
                """Perform a forward pass through the model."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])


    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        if getattr(self, "end2end", False):
            LOGGER.warning(
                "WARNING ⚠️ End2End model does not support 'augment=True' prediction. "
                "Reverting to single-scale prediction."
            )
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        nl = self.seqential_model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else DetectionLoss(self)


class OBBModel(DetectionModel):
    """YOLOv8 Oriented Bounding Box (OBB) model."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 OBB model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True): #刚开始都不传用默认值
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = create_model_dict(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.model_dict = cfg if isinstance(cfg, dict) else create_model_dict(cfg)  # cfg dict

        # Define model
        ch = self.model_dict["ch"] = self.model_dict.get("ch", ch)  # input channels
        if nc and nc != self.model_dict["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.model_dict['nc']} with nc={nc}")
            self.model_dict["nc"] = nc  # override YAML value
        elif not nc and not self.model_dict.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.seqential_model, self.save = parse_model(deepcopy(self.model_dict), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.model_dict["nc"])}  # default names dict
        self.model_info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(nn.Linear)  # last nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(nn.Conv2d)  # last nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        init_criterion: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):

        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss
        # 初始化模型中的损失类
        return RTDETRDetectionLoss(nc=self.nc)
    #RTDETRDetectionModel计算损失模块   算出预测和真实放入损失计算的到损失
    def loss(self, batch, preds=None):#RTDETRDetectionModel

        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion() 

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        img_idx = batch["img_idx"]
        bboxs_each_img = [(img_idx == i).sum().item() for i in range(bs)] #[4, 7]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "img_idx": img_idx.to(img.device, dtype=torch.long).view(-1),
            "bboxs_each_img": bboxs_each_img,
        }

        preds = self.predict(img, targets=targets) if preds is None else preds # imp
        dec_bboxes, dec_cls, topk_bboxes, topk_cls, dn_meta = preds if self.training else preds[1] #dec_bboxes, dec_cls, topk_bboxs, topk_cls, dn_meta
        if dn_meta is None:
            dn_bboxes, dn_cls = None, None #query
        else: #torch.Size([6, 4, 198, 4]) torch.Size([6, 4, 300, 4])
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_query_split"], dim=2) #torch.Size([6, 4, 198, 80]) torch.Size([6, 4, 300, 80])
            dn_cls, dec_cls = torch.split(dec_cls, dn_meta["dn_query_split"], dim=2)

        dec_bboxes = torch.cat([topk_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_cls = torch.cat([topk_cls.unsqueeze(0), dec_cls])

        loss = self.criterion(
            (dec_bboxes, dec_cls), targets, dn_bboxes=dn_bboxes, dn_cls=dn_cls, dn_meta=dn_meta
        )
        # NOTE: There are like 12 avg_loss_items in RTDETR, backward with all avg_loss_items but only show the main three avg_loss_items.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, targets=None, profile=False, visualize=False, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visualize (bool, optional): If True, save feature maps for visualization. Defaults to False.
            targets (dict, optional): Ground truth data for evaluation. Defaults to None.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.seqential_model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.seqential_model[-1]
        x = head([y[j] for j in head.f], targets)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model."""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters."""
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def generate_name_feats(self, name, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model."""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        if (
            not getattr(self, "clip_model", None) and cache_clip_model
        ):  # for backwards compatibility of models lacking clip_model attribute
            self.clip_model = clip.load("ViT-B/32")[0]
        clip_model = self.clip_model if cache_clip_model else clip.load("ViT-B/32")[0] #加载clip模型
        device = next(clip_model.parameters()).device
        name_token = clip.tokenize(name).to(device) # torch.Size([2, 77])
        name_feats = [clip_model.encode_text(token).detach() for token in name_token.split(batch)] #torch.Size([2, 512])
        name_feats = name_feats[0] if len(name_feats) == 1 else torch.cat(name_feats, dim=0) #torch.Size([2, 512])
        name_feats = name_feats / name_feats.norm(p=2, dim=-1, keepdim=True) #torch.Size([2, 512])
        self.txt_feats = name_feats.reshape(-1, len(name), name_feats.shape[-1]) #torch.Size([1, 2, 512]) 多少批 每一批多少 维度    #极重要
        self.seqential_model[-1].nc = len(name) #这个模型一个致命的缺点是得假设clip模型特别强

    def predict(self, x, profile=False, visualize=False, name_feats=None, augment=False, embed=None):

        if name_feats is None:
            if hasattr(self, 'name_feats'):
                name_feats = self.txt_feats  # 如果存在 name_feats 属性，使用它
            elif hasattr(self, 'txt_feats'):
                name_feats = self.txt_feats  # 如果存在 txt_feats 属性，使用它
            else:
                raise AttributeError("Neither 'name_feats' nor 'txt_feats' exists on the object.")

        name_feats = name_feats.to(device=x.device, dtype=x.dtype)
        if len(name_feats) != len(x): #对其
            name_feats = name_feats.repeat(len(x), 1, 1)
        ori_name_feats = name_feats.clone() #我觉得完全没必要克隆
        y, dt, embeddings = [], [], []  # outputs
        for m in self.seqential_model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, name_feats)
            elif isinstance(m, ImagePoolingAttn):
                name_feats = m(x, name_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_name_feats) #文本特征
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None): #重载损失

        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], name_feats=batch["name_feats"]) #前向传播是图像和文本特征
        return self.criterion(preds, batch)


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Example:
        ```python
        with temporary_modules({'old.module': 'new.module'}, {'old.module.attribute': 'new.module.attribute'}):
            import old.module  # this will now import new.module
            from old.module import attribute  # this will now import new.module.attribute
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """

    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def load_download_model(model_pt):
    """
    Attempts to load a PyTorch model with torch.load(). If ModuleNotFoundError is raised, installs the missing module
    and retries the load. Handles specific cases for YOLO models.

    Args:
        model_pt (str): File path of the PyTorch model.

    Returns:
        (dict): Loaded PyTorch model and model path.
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=model_pt, suffix=".pt")
    model_pt = attempt_download_asset(model_pt)  # Ensure local model file

    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",
            },
        ):
            ckpt = torch.load(model_pt, map_location="cpu")

    except ModuleNotFoundError as e:
        if e.name == "models":
            raise TypeError(emojis(
                f"ERROR ❌️ {model_pt} appears to be a YOLOv5 model incompatible with YOLOv8."
                f"\nTrain a new model with 'ultralytics' or use an official model (e.g., 'yolov8n.pt')."
            )) from e
        LOGGER.warning(
            f"WARNING ⚠️ {model_pt} requires '{e.name}', which is missing. Attempting auto-install."
        )
        check_requirements(e.name)
        ckpt = torch.load(model_pt, map_location="cpu")

    if not isinstance(ckpt, dict):
        LOGGER.warning(
            f"WARNING ⚠️ '{model_pt}' may be improperly formatted. For best results, use model.save('filename.pt')."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, model_pt



def attempt_load_weights(model_pts, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    ensemble = Ensemble()
    for model_pt in model_pts if isinstance(model_pts, list) else [model_pts]:
        ckpt, model_pt = load_download_model(model_pt)  # load ckpt
        train_args = {**DEFAULT_CFG_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None  # combined args
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = train_args  # 只要训练参数
        model.model_name = model_pt  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval())  # model in eval mode

    # Module updates
    for m in ensemble.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f"Ensemble created with {model_pts}\n")
    for k in "names", "nc", "yaml":
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"
    return ensemble


def attribute_assignment( model_pt, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, model_pt = load_download_model(model_pt)  # load ckpt  下载模型
   
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))} 
    # 训练参数多 模型参数少
    model.model_name = model_pt  # attach *.pt file path to model
    model.task  = guess_model_task(model)
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return ckpt, model


def parse_model(model_dict, ch, verbose=True):  # 通道是为了深拷贝
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (model_dict.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (model_dict.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = model_dict.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale] #深度（多高（多少个）） 最大通道数

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(model_dict["backbone"] + model_dict["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args): #遍历参数
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            
            ALSS,
            LCA,
            CA
        }:
            c1, c2 = ch[f], args[0] #输入 输出
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8) #成宽超参数
                
                
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]] # imp 总结 输入前一层 输出 第一个参数成宽度超参数 剩下的就是后面的参数
            
            
            if m in {
                BottleneckCSP,
                C1,
                C2,
                C2f,
                C3k2,
                C2fAttn,
                C3,
                C3TR,
                C3Ghost,
                C3x,
                RepC3,
                C2fPSA,
                C2fCIB,
                C2PSA,
                
                
                ALSS,
            }:

                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
            
        elif m is ECAAttention: #复现
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, *args[1:]]     
        
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}: #最后一层信息
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in {MSCAM,MSCAMv2,MSCAMv3,MSCAMv4,MSCAMv5}:
            c1 = c2 = ch[f]
            args = [c1, args[0]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 序列化的模型
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # 模型类
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i+1:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def create_model_dict(model_yaml):
    """Load a YOLOv8 model from a YAML file."""
    import re

    model_path = Path(model_yaml)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(model_path))  # i.e. yolov8x.yaml -> yolov8.yaml
    try:
        model_yaml = check_yaml(model_path)
    except Exception as e:
        print(f"Error parsing {model_path}: {e}")
        model_yaml = check_yaml(unified_path, hard=False)

    model_dict = yaml_load(model_yaml)  # model dict
    if "scale" not in model_dict or model_dict["scale"] is None:
        model_dict["scale"] = guess_model_scale(model_path)
    model_dict["model_name"] = str(model_path) #加入文件路径
    return model_dict


def guess_model_scale(model_path):

    with contextlib.suppress(AttributeError):
        import re

        return re.search(r"v\d+([nslmx])", Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ""


def guess_model_task(model): #实际模型或者字典
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(model_dict):
        """Guess from YAML dictionary."""
        task = model_dict["head"][-1][-2].lower()  # output module name 根据head-1-2猜测任务
        if task in ("classify", "classifier", "cls", "fc"):
            return "classify"
        if task == "detect":
            return "detect"
        if task == "segment":
            return "segment"
        if task == "pose":
            return "pose"
        if task == "obb":
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model 
        for x in "model.args", "model.model.args", "model.model.model.args": #第一个就是true
            with contextlib.suppress(Exception):
                return eval(x)["task"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
