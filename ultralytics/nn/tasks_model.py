# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import *
from ultralytics.nn.my_modules import *
from ultralytics.nn.extra_modules import *
from ultralytics.nn.reproduction_modules import *

from ultralytics.utils import DEFAULT_PARAM_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import v8ClassificationLoss, Anchor_Free_Detection_Loss, v8OBBLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.utils.plotting import visual_feature_map
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
from timm import create_model

class Base_Model(nn.Module):
    """图片直接预测，字典计算损失值."""

    def forward(self, batch_labels_list, *args, **kwargs): #成3了

        if isinstance(batch_labels_list, dict):  
            return self.computer_loss(batch_labels_list, *args, **kwargs) 
        return self.predict(batch_labels_list, *args, **kwargs)  #图片数据直接预测

    def computer_loss(self, batch_labels_list, preds=None):

        if not hasattr(self, "criterion"):
            self.loss_class = self.build_loss_class()

        preds = self.forward(batch_labels_list["img"]) if preds is None else preds #torch.Size([16, 144, 80, 80]) torch.Size([16, 144, 40, 40]) torch.Size([16, 144, 20, 20])
        return self.loss_class(preds, batch_labels_list)



    def build_loss_class(self):
        """Initialize the loss criterion for the Base_Model."""
        raise NotImplementedError("compute_loss() needs to be implemented by task_name heads")
    
    def predict(self, x, profile=False, visual_save_path=False, augment=False, embed=None):

        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visual_save_path, embed) #FFN

    def _predict_once(self, x, profile=False, visual_save_path=False, embed=None): # 前向传播一次

        y, dt, embeddings = [], [], []  # outputs
        for m in self.seqential_model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            # if isinstance(x, list):
            #     for detect in x:
            #         print("detect: {}".format(detect.size()))
            # else:
            #     print(x.size())

            y.append(x if m.i in self.save else None)  # 保留这一层输出
            if visual_save_path:
                visual_feature_map(x, m.type, m.i, save_path=visual_save_path)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"WARNING ⚠️ {self.__class__.__name__} does not support augmented inference yet. "
            f"Reverting to single-scale inference instead."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
 
        c = m == self.seqential_model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
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
            self.info(verbose=verbose)

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

    def info(self, detailed=False, verbose=True, imgsz=640):

        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz, ptflops=self.ptflops)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (Base_Model): An updated Base_Model object.
        """
        self = super()._apply(fn)
        m = self.seqential_model[-1]  # Detect()
        if isinstance(m, (Detect, Detect_DyHead, Detect_AFPN_P2345, Detect_AFPN_P2345_Custom, Detect_AFPN_P345, Detect_AFPN_P345_Custom, 
                          Detect_Efficient, DetectAux, Detect_DyHeadWithDCNV3, Segment, Segment_Efficient,Detect_SMART)):
        # if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
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





class Detection_Model(Base_Model): #检测模型
    """YOLOv8 detection model."""

    def __init__(self, model_dict, ch=3, nc=None, verbose=False, ptflops=False):
        super().__init__()

        # 将模型配置转换为字典
        self.model_dict = model_dict 
        self.ptflops= ptflops
        # 定义模型
        self._build_model(ch, nc, verbose) #输入输出

        # 初始化权重和偏置
        initialize_weights(self)

        # 输出模型信息
        if verbose:
            self.info()
            LOGGER.info("")

    def _build_model(self, ch, nc, verbose):
        self.model_dict["ch"] = ch #逻辑修改，使用传进来的 否则使用默认
        if nc and nc != self.model_dict["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.model_dict['nc']} with nc={nc}")
            self.model_dict["nc"] = nc #通道安装输入 类别按照输入但是model——dict可以自带

        # 创建模型并保存器
        self.seqential_model, self.save = parse_model(deepcopy(self.model_dict), ch=ch, verbose=verbose)

        # 构建步长
        last_layer_model = self.seqential_model[-1]
        # if isinstance(last_layer_model, Detect):        
        if isinstance(last_layer_model, (Detect, Detect_DyHead, Detect_AFPN_P2345, Detect_AFPN_P2345_Custom, Detect_AFPN_P345, Detect_AFPN_P345_Custom, 
                          Detect_Efficient, DetectAux, Detect_DyHeadWithDCNV3, Segment, Segment_Efficient, Pose,
                          Detect_SMART)):
            s = 640
            last_layer_model.inplace = self.model_dict.get("inplace", True)
            # forward = lambda x: self.forward(x)[0] if isinstance(last_layer_model, (Segment, Pose, OBB)) else self.forward(x,visual_save_path=Path("improved"))
            forward = lambda x: self.forward(x)[0] if isinstance(last_layer_model, (Segment, Pose, OBB)) else self.forward(x) #mark 可视化显示特征层
            self.stride = last_layer_model.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(2, ch, s, s))]) #1 3 256 256
            last_layer_model.bias_init()
        else:
            self.stride = torch.Tensor([32])





    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
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

    def build_loss_class(self):
        """Initialize the loss criterion for the Detection_Model."""
        return Anchor_Free_Detection_Loss(self)


class OBBModel(Detection_Model):
    """YOLOv8 Oriented Bounding Box (OBB) model."""

    def __init__(self, cfg="yolov8n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 OBB model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def build_loss_class(self):
        """Initialize the loss criterion for the model."""
        return v8OBBLoss(self)


class SegmentationModel(Detection_Model):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg="yolov8n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def build_loss_class(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return v8SegmentationLoss(self)


class PoseModel(Detection_Model):
    """YOLOv8 pose model."""

    def __init__(self, cfg="yolov8n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = creat_model_dict_add(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def build_loss_class(self):
        """Initialize the loss criterion for the PoseModel."""
        return v8PoseLoss(self)


class ClassificationModel(Base_Model):
    """YOLOv8 classification model."""

    def __init__(self, cfg="yolov8n-cls.yaml", ch=3, nc=None, verbose=True):
        """Init ClassificationModel with YAML, channels, number of classes, verbose flag."""
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.model_dict = cfg if isinstance(cfg, dict) else creat_model_dict_add(cfg)  # cfg dict

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
        self.info()

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
                i = types.index(nn.Linear)  # nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)  # nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def build_loss_class(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(Detection_Model):
    """
    RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    Detection_Model base class.

    Attributes:
        cfg (str): The configuration file path or preset string. Default is 'rtdetr-l.yaml'.
        ch (int): Number of input channels. Default is 3 (RGB).
        nc (int, optional): Number of classes for object detection. Default is None.
        verbose (bool): Specifies if summary statistics are shown during initialization. Default is True.

    Methods:
        build_loss_class: Initializes the criterion used for loss calculation.
        loss: Computes and returns the loss during training.
        predict: Performs a forward pass through the network and returns the output.
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """
        Initialize the RTDETRDetectionModel.

        Args:
            cfg (str): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes. Defaults to None.
            verbose (bool, optional): Print additional information during initialization. Defaults to True.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def build_loss_class(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.projects.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """
        Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (torch.Tensor, optional): Precomputed model predictions. Defaults to None.

        Returns:
            (tuple): A tuple containing the total loss and main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.loss_class = self.build_loss_class()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.loss_class(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visual_save_path=False, batch=None, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visual_save_path (bool, optional): If True, save feature maps for visualization. Defaults to False.
            batch (dict, optional): Ground truth data for evaluation. Defaults to None.
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
            if visual_save_path:
                visual_feature_map(x, m.type, m.i, save_path=visual_save_path)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.seqential_model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(Detection_Model):
    """YOLOv8 World Model."""

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters."""
        self.txt_feats = torch.randn(1, nc or 80, 512)  # placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text):
        """Perform a forward pass with optional profiling, visualization, and embedding extraction."""
        try:
            import clip
        except ImportError:
            check_requirements("git+https://github.com/openai/CLIP.git")
            import clip

        model, _ = clip.load("ViT-B/32")
        device = next(model.parameters()).device
        text_token = clip.tokenize(text).to(device)
        txt_feats = model.encode_text(text_token).to(dtype=torch.float32)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        self.txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1]).detach()
        self.seqential_model[-1].nc = len(text)

    def build_loss_class(self):
        """Initialize the loss criterion for the model."""
        raise NotImplementedError

    def predict(self, x, profile=False, visual_save_path=False, augment=False, embed=None):
        """
        Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool, optional): If True, profile the computation time for each layer. Defaults to False.
            visual_save_path (bool, optional): If True, save feature maps for visualization. Defaults to False.
            augment (bool, optional): If True, perform data augmentation during inference. Defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = self.txt_feats.to(device=x.device, dtype=x.dtype)
        if len(txt_feats) != len(x):
            txt_feats = txt_feats.repeat(len(x), 1, 1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        for m in self.seqential_model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visual_save_path:
                visual_feature_map(x, m.type, m.i, save_path=visual_save_path)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visual_save_path=False):
        """Function generates the YOLO network's final layer."""
        y = [module(x, augment, profile, visual_save_path)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the false import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping false module paths to true module paths.

    Example:
        ```python
        with temporary_modules({'false.module.path': 'true.module.path'}):
            import false.module.path  # this will now import true.module.path
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if not modules:
        modules = {}

    import importlib
    import sys

    try:
        # Set modules in sys.modules under their false name
        for false, true in modules.items():
            sys.modules[false] = importlib.import_module(true)

        yield
    finally:
        # Remove the temporary module paths
        for false in modules:
            if false in sys.modules:
                del sys.modules[false]


def generate_ckpt(model_pt):

    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=model_pt, suffix=".pt")
    model_pt = attempt_download_asset(model_pt)  # 'yolov8n.pt'
    try:
        with temporary_modules(
            {
                "ultralytics.yolo.utils": "ultralytics.utils", #第三方工具
                "ultralytics.yolo.v8": "ultralytics.projects.yolo", #大模型
                "ultralytics.yolo.data": "ultralytics.data", #数据处理
            } #全部映射成功
        ):  # for legacy 8.0 Classify and Pose models
            ckpt = torch.load(model_pt, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {model_pt} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a true model using the latest 'ultralytics' package or to "
                    f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'"
                )
            ) from e
        LOGGER.warning(
            f"WARNING ⚠️ {model_pt} appears to require '{e.name}', which is not in ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a true model using the latest 'ultralytics' package or to "
            f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch.load(model_pt, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"WARNING ⚠️ The weight_str '{model_pt}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, model_pt  # load


def attempt_load_weights(model_pts, device=None, inplace=True, fuse=False):
    """Loads models or an ensemble of models from weights."""

    # Prepare an ensemble
    ensemble = Ensemble()

    # Ensure model_pts is a list
    model_pts = model_pts if isinstance(model_pts, list) else [model_pts]

    # Load each model
    for model_pt in model_pts:
        # Generate checkpoint
        ckpt, model_pt = generate_ckpt(model_pt)

        # Get training arguments
        args = {**DEFAULT_PARAM_DICT, **ckpt["train_args"]} if "train_args" in ckpt else None

        # Load model
        model = (ckpt.get("ema") or ckpt["model"]).to(device).float()
        model.args = args  # attach args
        model.model_str = model_pt  # attach path
        model.task_name = creat_model_task_name(model)  # attach task name

        # Compatibility checks
        if not hasattr(model, "stride"):
            model.stride = torch.tensor([32.0])

        # Append to ensemble
        if fuse and hasattr(model, "fuse"):
            ensemble.append(model.fuse().eval())  # Fuse and set to eval mode
        else:
            ensemble.append(model.eval())  # Set to eval mode

    # Module updates
    for m in ensemble.modules():
        t = type(m)
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # Compatibility for torch 1.11.0

    # If only one model is loaded, return the model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Otherwise, return an ensemble
    # Set shared attributes for the ensemble
    for attr in ["names", "nc", "yaml"]:
        setattr(ensemble, attr, getattr(ensemble[0], attr))
    ensemble.stride = ensemble[int(torch.argmax(torch.tensor([m.stride.max() for m in ensemble])))].stride

    # Ensure all models have the same class count
    assert all(ensemble[0].nc == m.nc for m in ensemble), f"Models differ in class counts {[m.nc for m in ensemble]}"

    LOGGER.info(f"Ensemble created with models: {[m.model_str for m in ensemble]}")
    return ensemble



def load_pytorch_model(model_pt, device=None, inplace=True, fuse=False): #'yolov8n.pt'
    """Loads a single model weights."""
    ckpt, model_pt = generate_ckpt(model_pt)  # dict   'yolov8n.pt'               load ckpt  
    args = {**DEFAULT_PARAM_DICT, **(ckpt.get("train_args", {}))}  # 优先模型参数                combine default args and model, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.model_str = model_pt  #str->str                         attach *.pt file path to model
    model.task_name = creat_model_task_name(model)    #'detect'
    if not hasattr(model, "stride"): 
        model.stride = torch.tensor([32.0])

    model = model.fuse().eval() if fuse and hasattr(model, "fuse") else model.eval()  # model in eval mode评估模式

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):#是上采样并且没有recompute_scale_factor这个成员属性
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(model_dict, ch, verbose=True):  # model_dict, input_channels(3)  模型序列化
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    max_channels = float("inf")
    depth, width, kpt_shape = (model_dict.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")) #0.33 0.25 1.0
    nc, act, scales = (model_dict.get(x) for x in ("nc", "activation", "scales")) #80 none none  通道 激活 尺度

    if scales:
        scale = model_dict.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"警告 ⚠️ 没有传入模型具体尺度（包括模型深度（堆叠层数）和宽度（单层通道数目））. 猜测的尺度是'{scale}'.") #使用第一个尺度
        depth, width, max_channels = scales[scale] #尺度中给出了三个值 深度 宽度 最大通道数
    if act:
        Conv.default_act = eval(act)  #设置卷积的默认激活函数
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # 打印激活函数
    if verbose:  #打印模型信息
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    ch = [ch]  #[3]
    model, save, c2 = [], [], []  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(model_dict["backbone"] + model_dict["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        # 深层 宽通
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain   1
        if m in (Classify,Conv,ConvTranspose,GhostConv,Bottleneck,GhostBottleneck,SPP,SPPF,DWConv,Focus,BottleneckCSP,C1,
            C2,C2f,RepNCSPELAN4,ADown,SPPELAN,C2fAttn,C3,C3TR,C3Ghost,nn.ConvTranspose2d,DWConvTranspose2d,C3x,RepC3,


            # 改进
            GSConv, VoVGSCSP, VoVGSCSPC,
            PConv,
            ShuffleNetV2,
            ShuffleNetV3,
            BottleneckCSP,
            CoordAttv2,
            ASSNET,
            CoordAtt, #原来是有歧义
            CSPStage,

        ):
            c1, c2 = ch[f], args[0] #给出输入 输出通道


            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8) #输出通道的操作

            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]



            if m in (BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, VoVGSCSP, VoVGSCSPC,
                      ASSNET, CSPStage                  ):
                args.insert(2, n)  # 插入个数个数变成1   number of repeats
                n = 1


        elif m is BiLevelRoutingAttention:#复现
            c2 = ch[f]
            args = [c2, *args]

        elif m is ECAAttention: #复现
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, *args[1:]]


        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
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



        # elif m in (Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn):
        elif m in (Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, 
                   Detect_DyHead, Detect_AFPN_P2345, Detect_AFPN_P2345_Custom, Detect_AFPN_P345, Detect_AFPN_P345_Custom, 
                   Detect_Efficient, DetectAux, Detect_DyHeadWithDCNV3, Segment_Efficient,
                    Detect_SMART):
            # 添加输入通道
            args.append([ch[x] for x in f])

        elif m in (iAFF, AFF):
            channels = [ch[x] for x in f]
            c2 = channels[0]       # output of the iAFF module ( output channel is eighter of the channels)
            args = [c2]



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

        # 改进
        elif m is space_to_depth:
            c2 = 4 * ch[f]    
        else:
            c2 = ch[f]

        layer_model = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # 
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in layer_model.parameters())  # number params
        layer_model.i, layer_model.f, layer_model.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        model.append(layer_model)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*model), sorted(save)


def creat_model_dict_add(model_str):
    import re
    model_str = Path(model_str)
    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(model_str))  # i.e. yolov8x.yaml -> yolov8.yaml
    model_cfg2 = check_yaml(unified_path, hard=False) or check_yaml(model_str) #优先使用通用的
    model_dict = yaml_load(model_cfg2)  # model dict
    model_dict["scale"] = creat_model_scale(model_str) #n
    model_dict["data_str"] = str(model_str)
    return model_dict


def creat_model_scale(model_path):
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r"yolov\d+([nslmx])", Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ""


def creat_model_task_name(model): #class{detectionmodel} 只取了模型部分

    def model_to_task(model_dict):
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
            return model_to_task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task_name"]
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return model_to_task(eval(x))

        for m in model.modules():
            if isinstance(m, Segment):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            # if isinstance(m, (Detect, Detect_DyHead, Detect_AFPN_P2345, Detect_AFPN_P2345_Custom, 
            #                   Detect_AFPN_P345, Detect_AFPN_P345_Custom, Detect_Efficient, DetectAux,
            #                   Detect_DyHeadWithDCNV3,WorldDetect,GhostConvDetect,GhostConvDetect)):
            elif isinstance(m, (Detect, WorldDetect)):            
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

    # Unable to determine task_name from model
    LOGGER.warning(
        "WARNING ⚠️ Unable to automatically guess model task_name, assuming 'task_name=detect'. "
        "Explicitly define task_name for your model, i.e. 'task_name=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect



if __name__ == "__main__":
    # model_str = "ultralytics/cfg_yaml/model_yaml/test_model_yaml/ShuffleNetV3.yaml"
    # model_str = "ultralytics/cfg_yaml/论文消融实验模型/light-focus-poolconv-atta.yaml"
    model_str = "ultralytics/cfg_yaml/论文消融实验模型2/ASS-newfocus.yaml"
    # model_str = "ultralytics/cfg_yaml/论文消融实验模型/light.yaml"
    # model_str = "ultralytics/cfg_yaml/test_model_yaml/ShuffleNetV4-nofocus.yaml"
    
    # model_str = "runs/detect/v8_fore_detect/fore_detetct.yaml"
    # model_str = "ultralytics/cfg_yaml/model_yaml/test_model_yaml/PConv3.yaml"
    ch =1
    model_dict = creat_model_dict_add(model_str)
    # from ultralytics.nn.tasks import Detection_Model
    detection_model = Detection_Model(model_dict, ch=1, nc=4, verbose=True,ptflops=False)
    detection_model.fuse()
    x=torch.zeros(1, ch, 64, 64)
    detection_model._predict_once(x)

# def get_flops(model, imgsz=640):

    # [10, [64, 128, 256]]   [10, [192, 256, 392]] 

    # Model summary: 225 layers, 2539566 parameters, 2539550 gradients
    # Model summary: 191 layers, 2549573 parameters, 2549557 gradients

#      22        [15, 18, 21]  1    753262  ultralytics.nn.common.head.Detect            [10, [64, 128, 256]]          
# Model summary: 225 layers, 2539566 parameters, 2539550 gradients, 7.0 GFLOPs
# Model summary: 199 layers, 1384020 parameters, 1384004 gradients