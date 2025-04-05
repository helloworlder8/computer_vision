# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect"


class Detect(nn.Module):#检测头
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80,reg_max=16, ch=()): #[256, 512, 1024]
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        if isinstance(reg_max, list) and len(reg_max) > 0:
            ch = reg_max
            reg_max = 16
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # 两个中间通道
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        ) #4 * self.reg_max最后通道 
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch) #self.nc最后通道 分类
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl): #只玩通道 CV2算回归 CV3算分类
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) #-》[torch.Size([2, 144, 80, 80]), torch.Size([2, 144, 40, 40]), torch.Size([2, 144, 20, 20])] 回归 类别
        if self.training:  # Training path  144 的话包含80分类和64目标检测
            return x
        y = self._inference(x) #torch.Size([2, 84, 3528])
        return y if self.export else (y, x) #torch.Size([1, 84, 8400]) 以及每一层输入

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one predn.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one predn separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x): # [torch.Size([2, 144, 80, 80]), torch.Size([2, 144, 40, 40]), torch.Size([2, 144, 20, 20])]
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  #  torch.Size([2, 144, 80, 80])
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) #torch.Size([2, 144, 8400])
        if self.dynamic or self.shape != shape:
            self.anc_points, self.stride_tensor = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) #->torch.Size([2, 64, 8400]) torch.Size([2, 80, 8400])

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.stride_tensor / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anc_points.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anc_points.unsqueeze(0)) * self.stride_tensor #torch.Size([1, 4, 8400])
            # torch.Size([1, 4, 8400]) torch.Size([1, 2, 8400])
        return torch.cat((dbox, cls.sigmoid()), 1) #torch.Size([2, 4, 8400]) torch.Size([2, 80, 8400])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors): #torch.Size([2, 4, 8400]) torch.Size([1, 2, 8400])
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes the predictions obtained from a YOLOv10 model.

        Args:
            preds (torch.Tensor): The predictions obtained from the model. It should have a shape of (batch_size, num_boxes, 4 + nc).
            max_det (int): The maximum number of predn to keep.
            nc (int, optional): The number of classes. Defaults to 80.

        Returns:
            (torch.Tensor): The post-processed predictions with shape (batch_size, max_det, 6),
                including bounding boxes, scores and cls.
        """
        assert 4 + nc == preds.shape[-1]
        boxes, scores = preds.split([4, nc], dim=-1)
        max_scores = scores.amax(dim=-1)
        max_scores, index = torch.topk(max_scores, min(max_det, max_scores.shape[1]), axis=-1)
        index = index.unsqueeze(-1)
        boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
        scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

        # NOTE: simplify but result slightly lower mAP
        # scores, labels = scores.max(dim=-1)
        # return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

        scores, index = torch.topk(scores.flatten(1), max_det, axis=-1)
        labels = index % nc
        index = index // nc
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

        return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)


class Segment(Detect):
    """YOLOv8 Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256,reg_max = 16, ch=()): #[80, 32, 256, [256, 512, 1024]]
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        if isinstance(reg_max, list) and len(reg_max) > 0:
            ch = reg_max
            reg_max = 16
        super().__init__(nc,reg_max,ch) #最后才加输入通道信息
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos #直接理解成中间层
        self.proto = Proto(ch[0], self.npr, self.nm)  # 输入 中间 输出掩膜数

        c4 = max(ch[0] // 4, self.nm) #值等于第一个层通道除以4 中间输出通道    self.nm最后输出通道
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x): #[torch.Size([2, 64, 80, 80]), torch.Size([2, 128, 40, 40]), torch.Size([2, 256, 20, 20])]
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        proto = self.proto(x[0])  # mask protos  卷积 上采样 卷积 卷积 -> torch.Size([2, 32, 160, 160])
        bs = proto.shape[0]  # batch size  2     80x80 40x40 20x20
        # [torch.Size([2, 256, 80, 80]), torch.Size([2, 512, 40, 40]), torch.Size([2, 1024, 20, 20])]
        Mask_Coeff = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  #torch.Size([2, 32, 8400]) 通道统一到32
        x = Detect.forward(self, x)
        if self.training:
            # feats, Mask_Coeff, proto
            return x, Mask_Coeff, proto #-》[torch.Size([2, 144, 80, 80]), torch.Size([2, 144, 40, 40]), torch.Size([2, 144, 20, 20])]
        return (torch.cat([x, Mask_Coeff], 1), proto) if self.export else (torch.cat([x[0], Mask_Coeff], 1), (x[1], Mask_Coeff, proto)) #第一个是解析后的数值 第二个是原始材料
        #  回归加掩码
# Mask_Coeff
class OBB(Detect):
    """YOLOv8 OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLOv8 Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
        padding, and groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class WorldDetect(Detect):
    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLOv8 detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100)) #128
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch) #不管什么通道转换成嵌入维度
        
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
    # torch.Size([4, 128, 80, 80]) torch.Size([4, 256, 40, 40]) torch.Size([4, 512, 20, 20])  torch.Size([4, 80, 512])
    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1) #回归信息 分类信息中强制加入文本特征编码然后进行融合 #64 是回归 80是token 合并成144
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.nq = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.dn_embed = nn.Embedding(nc, hd) #类别 影藏层
        self.num_dn = nd
        self.cls_dn_coef = label_noise_ratio
        self.bboxs_dn_coef = box_noise_scale
# dn
        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_linear = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()
    # torch.Size([2, 256, 80, 80]) torch.Size([2, 256, 40, 40]) torch.Size([2, 256, 20, 20])
    def forward(self, feats, gt=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import generate_cdn_train_sample
        if feats[0].shape[0]==2: #RTDETRDecoder调试
            print("train")
        
        # Input projection and embedding
        feats, shapes = self._feats_converse(feats) #torch.Size([2, 8400, 256])  [[80, 80], [40, 40], [20, 20]]
        # 准备去噪训练
        # Prepare denoising training  torch.Size([4, 192, 256]) torch.Size([4, 192, 4]) torch.Size([492, 492])   [torch.Size([48]) torch.Size([96]) torch.Size([84]) torch.Size([36])] 12  [192, 300]
        dn_cls_embed, dn_bboxs, attn_mask, dn_meta = generate_cdn_train_sample(
            gt, #对目标进行处理
            self.nc, #80
            self.nq, #300
            self.dn_embed.weight, #torch.Size([80, 256])
            self.num_dn, #num denoising噪声数 100
            self.cls_dn_coef, #0.5
            self.bboxs_dn_coef, #1.0
            self.training, #true
        )
# combined_cls_embed, combined_bboxs, enc_bboxes, cls_value topk_cls topk_bboxs
        combined_cls_embed, combined_bboxs, topk_bboxs, topk_cls = self._generate_combined_cls_bboxs(feats, shapes, dn_cls_embed, dn_bboxs)

        # Decoder
        dec_bboxes, dec_cls = self.decoder( #torch.Size([6, 2, 492, 4]) torch.Size([6, 2, 492, 80])
            combined_cls_embed,
            combined_bboxs,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        ) #torch.Size([6, 2, 496, 4]) torch.Size([6, 2, 496, 80]) torch.Size([2, 300, 4]) torch.Size([2, 300, 80])
        x = dec_bboxes, dec_cls, topk_bboxs, topk_cls, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_cls.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy_norm = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy_norm, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy_norm, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask  #后面的有点用

    def _feats_converse(self, feats):#torch.Size([4, 256, 80, 80]) torch.Size([4, 256, 40, 40]) torch.Size([4, 256, 20, 20])
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]#-》torch.Size([4, 256, 80, 80]) torch.Size([4, 256, 40, 40]) torch.Size([4, 256, 20, 20]) 线性变化
        # Get encoder inputs
        feat_list = []
        shapes = []
        for feat in feats:
            h, w = feat.shape[2:] #80 80
            # [torch.Size([4, 6400, 256]), torch.Size([4, 1600, 256]), torch.Size([4, 400, 256])]
            feat_list.append(feat.flatten(2).permute(0, 2, 1)) #批 点 通
            # [[80, 80], [40, 40], [20, 20]]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feat_list, 1) #torch.Size([4, 8400, 256])
        return feats, shapes #最终的特征和对应的形状

    # def _generate_combined_cls_bboxs(self, feats, shapes, dn_cls_embed=None, dn_bboxs_embed=None):
    #     """Generates and prepares the input required for the decoder from the provided features and shapes."""
    #     bs = feats.shape[0]
    #     # Prepare input for decoder
    #     anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device) #torch.Size([1, 8400, 4]) torch.Size([1, 8400, 1])
    #     features = self.enc_linear(valid_mask * feats)  #torch.Size([4, 8400, 256])再经过一层线性变化

    #     cls_value = self.enc_score_head(features)  # torch.Size([4, 8400, 80])

    #     # Query selection
    #     # torch.Size([600])
    #     Topk = torch.topk(cls_value.max(-1).values, self.nq, dim=1).indices.view(-1) #每张图片中取300个置信度最高的
    #     # torch.Size([600])
    #     batch_index = torch.arange(end=bs, dtype=Topk.dtype).unsqueeze(-1).repeat(1, self.nq).view(-1)

    #     # torch.Size([2, 300, 256])  每张图片查询300个模式
    #     topk_cls_embed = features[batch_index, Topk].view(bs, self.nq, -1) 
    #     # torch.Size([2, 300, 4])
    #     topk_anchors = anchors[:, Topk].view(bs, self.nq, -1)

    #     # Dynamic anchors + static content
    #     topk_bboxs = self.enc_bbox_head(topk_cls_embed) + topk_anchors
    #     # torch.Size([4, 300, 4])
    #     enc_bboxes = topk_bboxs.sigmoid()
    #     if dn_bboxs_embed is not None:
    #         combined_bboxs = torch.cat([dn_bboxs_embed, topk_bboxs], 1)
    #     cls_value = cls_value[batch_index, Topk].view(bs, self.nq, -1)

    #     topk_cls_embed = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else topk_cls_embed #torch.Size([2, 300, 256])
    #     if self.training:
    #         combined_bboxs = combined_bboxs.detach()
    #         if not self.learnt_init_query:
    #             topk_cls_embed = topk_cls_embed.detach()
    #     if dn_cls_embed is not None:
    #         combined_cls_embed = torch.cat([dn_cls_embed, topk_cls_embed], 1)
    #     # top_k加噪声  top_k加噪声 torch.Size([4, 300, 4]) torch.Size([4, 300, 80])
    #     return combined_cls_embed, combined_bboxs, enc_bboxes, cls_value


    def _generate_combined_cls_bboxs(self, feats, shapes, dn_cls_embed=None, dn_bboxs=None):
        """Generates and prepares the input required for the decoder from the provided feats and shapes."""
        
        bs = feats.shape[0]
        
        """ 吹得贼牛逼的iou-aware """
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)  # torch.Size([1, 8400, 4]) torch.Size([1, 8400, 1])
        feats = self.enc_linear(valid_mask * feats)  # torch.Size([4, 8400, 256])再经过一层线性变化
        cls_value = self.enc_score_head(feats)  # torch.Size([4, 8400, 80])


        """ 类别 """
        # Query selection - selecting top k based on confidence scores
        Topk = torch.topk(cls_value.max(-1).values, self.nq, dim=1).indices.view(-1)  # 每张图片中取300个置信度最高的
        batch_index = torch.arange(end=bs, dtype=Topk.dtype).unsqueeze(-1).repeat(1, self.nq).view(-1)  # torch.Size([600])
        # Process classification scores
        topk_cls = cls_value[batch_index, Topk].view(bs, self.nq, -1)  # torch.Size([2, 300, 80])
        # Extract top-k feats and anchors for each image
        topk_cls_embed = feats[batch_index, Topk].view(bs, self.nq, -1)  # torch.Size([2, 300, 256])
        
        

        
        
        
        """ 回归 """
        topk_anchors = anchors[:, Topk].view(bs, self.nq, -1)  # torch.Size([2, 300, 4])
        # Dynamic anchors + static content
        topk_bboxs = self.enc_bbox_head(topk_cls_embed) + topk_anchors  # torch.Size([2, 300, 4])
        # Combine dynamic and noise-augmented bbox embeddings if provided
        combined_bboxs = torch.cat([dn_bboxs, topk_bboxs], 1) if dn_bboxs is not None else topk_bboxs
        
        
        topk_bboxs = topk_bboxs.sigmoid()  # torch.Size([4, 300, 4])


        # Initialize or modify the query embedding for classification
        if self.learnt_init_query:
            topk_cls_embed = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # torch.Size([2, 300, 256])

        # Detach during training if necessary
        if self.training:
            combined_bboxs = combined_bboxs.detach()
            if not self.learnt_init_query:
                topk_cls_embed = topk_cls_embed.detach()

        # Combine noise-augmented embeddings for classification if provided
        combined_cls_embed = torch.cat([dn_cls_embed, topk_cls_embed], 1) if dn_cls_embed is not None else topk_cls_embed

        # Return final combined results
        return combined_cls_embed, combined_bboxs, topk_bboxs, topk_cls








    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_linear[0])
        xavier_uniform_(self.enc_linear[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect): #yolov10检测头
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of predn.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)
