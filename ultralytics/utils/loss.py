# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors

from .metrics import bbox_iou, probiou, bbox_mpdiou, bbox_inner_iou, bbox_inner_mpdiou, wasserstein_loss, WiseIouLoss
from .tal import bbox2dist

import math
 
 
class SlideLoss(nn.Module):
    def __init__(self, loss_fcn):
        super(SlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element

    def forward(self, pred, true, auto_iou=0.5):
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class EMASlideLoss:
    def __init__(self, loss_fcn, decay=0.999, tau=2000):
        super(EMASlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        self.is_train = True
        self.updates = 0
        self.iou_mean = 1.0
    
    def __call__(self, pred, true, auto_iou=0.5):
        if self.is_train and auto_iou != -1:
            self.updates += 1
            d = self.decay(self.updates)
            self.iou_mean = d * self.iou_mean + (1 - d) * float(auto_iou.detach())
        auto_iou = self.iou_mean
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss



class QualityfocalLoss(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, pred_score, gt_score, gt_target_pos_mask):
        # negatives are supervised by 0 quality score
        pred_sigmoid = pred_score.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred_score.shape)
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(pred_score, zerolabel, reduction='none') * scale_factor.pow(self.beta)
        
        scale_factor = gt_score[gt_target_pos_mask] - pred_sigmoid[gt_target_pos_mask]
        with torch.cuda.amp.autocast(enabled=False):
            loss[gt_target_pos_mask] = F.binary_cross_entropy_with_logits(pred_score[gt_target_pos_mask], gt_score[gt_target_pos_mask], reduction='none') * scale_factor.abs().pow(self.beta)
        return loss
    

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


def dice_loss(pred, target, smooth=1e-5):
    # pred = torch.sigmoid(pred)
    intersection = (pred * target)
    union = pred + target
    dice_score = (2 * intersection + smooth) / (union + smooth)
    loss = 1 - dice_score
    return loss  # 返回与输入形状相同的损失

class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/avg_loss_items/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL avg_loss_items during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pd_dist, target):
        """
        Return sum of left and right DFL avg_loss_items.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01) #-》torch.Size([160, 4])
        tl = target.long()  # target left  torch.Size([160, 4])
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pd_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pd_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training avg_loss_items during training."""

    def __init__(self, reg_max=16, IoU_algorithm="CIoU"):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.IoU_algorithm = IoU_algorithm



    def forward(self, pd_dist, pd_bboxes_pyramid, anc_points, target_scores, target_bboxes_pyramid, target_scores_sum, TF_fg, mpdiou_hw=None):
        """IoU loss."""
        weight = target_scores.sum(-1)[TF_fg].unsqueeze(-1) #torch.Size([2, 8400, 80])

        IoU_options = {
            "GIoU": False,
            "DIoU": False,
            "CIoU": False,
            "EIoU": False,
            "SIoU": False,
            "FineSIoU": False,
            "WIoU": False
        }
        if self.IoU_algorithm in IoU_options:
            IoU_options[self.IoU_algorithm] = True
        else:
            raise ValueError(f"Unsupported IOU algorithm: {self.IoU_algorithm}")   
        iou = bbox_iou(pd_bboxes_pyramid[TF_fg], target_bboxes_pyramid[TF_fg], xywh=False, **IoU_options)
        if type(iou) is tuple:
            if len(iou) == 2:
                loss_iou = ((1.0 - iou[0]) * iou[1].detach() * weight).sum() / target_scores_sum
            else:
                loss_iou = (iou[0] * iou[1] * weight).sum() / target_scores_sum
        else:
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
                

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anc_points, target_bboxes_pyramid, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pd_dist[TF_fg].view(-1, self.dfl_loss.reg_max), target_ltrb[TF_fg]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pd_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training avg_loss_items during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pd_dist, pd_bboxes, anc_points, target_bboxes, target_scores, target_scores_sum, TF_fg):
        """IoU loss."""
        weight = target_scores.sum(-1)[TF_fg].unsqueeze(-1)
        iou = probiou(pd_bboxes[TF_fg], target_bboxes[TF_fg])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anc_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pd_dist[TF_fg].view(-1, self.dfl_loss.reg_max), target_ltrb[TF_fg]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pd_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training avg_loss_items."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class DetectionLoss:
    """Criterion class for computing training avg_loss_items."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.seqential_model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none') # 分类损失
        # self.bce = EMASlideLoss(nn.BCEWithLogitsLoss(reduction='none'))  # Exponential Moving Average Slide Loss
        # self.bce = SlideLoss(nn.BCEWithLogitsLoss(reduction='none')) # Slide Loss
        # self.bce = FocalLoss(alpha=0.25, gamma=1.5) # FocalLoss
        # self.bce = VarifocalLoss(alpha=0.75, gamma=2.0) # VarifocalLoss
        # self.bce = QualityfocalLoss(beta=2.0) # QualityfocalLoss
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, nc=self.nc, alpha=0.5, beta=6.0, IoU_algorithm=model.args.IoU)
        # self.assigner = TaskAlignedAssigner(topk=tal_topk, nc=self.nc, alpha=0.5, beta=6.0, IoU_algorithm="SIoU")
        self.bbox_loss = BboxLoss(m.reg_max, model.args.IoU).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pd_distri, pd_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        ) #批 维 点   批维点

        pd_scores = pd_scores.permute(0, 2, 1).contiguous() #torch.Size([4, 8400, 80])
        pd_distri = pd_distri.permute(0, 2, 1).contiguous() #torch.Size([4, 8400, 64])

        dtype = pd_scores.dtype
        batch_size = pd_scores.shape[0] #4
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anc_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["img_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1) #标签层面到真实标签层面
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2) #批 框 维 cls, xyxy
        gt_mask = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pd_bboxes_pyramid = self.bbox_decode(anc_points, pd_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, _, TF_fg = self.assigner( #标签分配器->torch.Size([2, 8400, 4]) torch.Size([2, 8400, 80])
            pd_scores.detach().sigmoid(), #->torch.Size([2, 8400])
            (pd_bboxes_pyramid.detach() * stride_tensor).type(gt_bboxes.dtype),
            anc_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            gt_mask,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pd_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pd_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE torch.Size([2, 8400, 80])

        # Bbox loss
        if TF_fg.sum():
            target_bboxes_pyramid = target_bboxes/stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pd_distri, pd_bboxes_pyramid, anc_points, target_scores, target_bboxes_pyramid, target_scores_sum, TF_fg, ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1, batch_size).transpose(1, 0)
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    
    
    # torch.Size([21, 6])  2   torch.Size([4])
    def preprocess(self, targets, batch_size, scale_tensor): #金字塔到计算机图像域  xywh-》xyxy
        """Preprocesses the target counts and matches with the input batch size to output a tensor.""" #num_dim
        num_tag, num_dim = targets.shape
        if num_tag == 0:
            gt_targets = torch.zeros(batch_size, 0, num_dim - 1, device=self.device)
        else:
            img_idx = targets[:, 0]  # image index  torch.Size([2]) torch.Size([2])
            _, counts = img_idx.unique(return_counts=True) #{Tensor:(21,)} tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0')  {Tensor:(2,)} tensor([ 8, 13], device='cuda:0', dtype=torch.int32)
            counts = counts.to(dtype=torch.int32)
            gt_targets = torch.zeros(batch_size, counts.max(), num_dim - 1, device=self.device) #t 2 13 5
            for j in range(batch_size):
                TF = img_idx == j
                n = TF.sum()
                if n:
                    gt_targets[j, :n] = targets[TF, 1:]
            gt_targets[..., 1:5] = xywh2xyxy(gt_targets[..., 1:5].mul_(scale_tensor)) #gt_targets 金字塔后的图像坐标
        return gt_targets
    def bbox_decode(self, anc_points, pd_dist): #8400 2       2 8400 64
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pd_dist.shape  # batch, anchors, channels
            pd_dist = pd_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pd_dist.dtype)) #2 8400 4
            # pd_dist = pd_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pd_dist.dtype))
            # pd_dist = (pd_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pd_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pd_dist, anc_points, xywh=False) #torch.Size([2, 8400, 4])   torch.Size([8400, 2])



def combined_loss(pred, target, alpha=0.5):
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")  # 保持与输入相同形状
    dice_loss_value = dice_loss(pred, target)
    combined_loss = alpha * bce_loss + (1 - alpha) * dice_loss_value
    return combined_loss  # 返回与输入形状相同的损失

class SegmentationLoss(DetectionLoss):
    """Criterion class for computing training avg_loss_items."""
    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap_mask = model.args.overlap_mask
        
    def __call__(self, preds, batch): #预测的掩膜带有置信度
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl    t4
        feats, pd_seg_matrix, proto = self.unpack_preds(preds) #[torch.Size([2, 144, 80, 80]), torch.Size([2, 144, 40, 40]), torch.Size([2, 144, 20, 20])]  torch.Size([2, 32, 8400])  torch.Size([2, 32, 160, 160])
        # feats充满了妥协
        pd_distri, pd_scores, pd_seg_matrix = self.prepare_predictions(feats, pd_seg_matrix) # ->批 点 维
        # torch.Size([2, 8400, 64]) torch.Size([2, 8400, 80]) torch.Size([2, 8400, 32])
        dtype = pd_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        batch_size, _, mask_h, mask_w = proto.shape # 2 160 160
        anc_points, stride_tensor = make_anchors(feats, self.stride, 0.5) #-》torch.Size([8400, 2]) torch.Size([8400, 1])
        
        gt_labels, gt_bboxes, gt_mask = self.generate_targets(batch, imgsz, batch_size)#-》torch.Size([2, 13, 1]) torch.Size([2, 13, 4]) torch.Size([2, 13, 1])
        
        pd_bboxes_pyramid = self.bbox_decode(anc_points, pd_distri) #-》torch.Size([2, 8400, 4]) 金字塔坐标
        # 这之后都是真实值坐标
        _, target_bboxes, target_scores, fg_inds, TF_fg = self.assigner( #-》torch.Size([2, 8400, 4]) torch.Size([2, 8400, 80]) torch.Size([2, 8400])  torch.Size([2, 8400])
            pd_scores.detach().sigmoid(),
            (pd_bboxes_pyramid.detach() * stride_tensor).type(gt_bboxes.dtype),
            anc_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            gt_mask,
        )
        
        target_scores_sum = max(target_scores.sum(), 1)
        # 分类损失  torch.Size([2, 8400, 80]) torch.Size([2, 8400, 80])
        loss[2] = self.bce(pd_scores, target_scores.to(dtype)).sum() / target_scores_sum
        
        if TF_fg.sum(): #bbox损失 dfl损失
            # loss[0], loss[3] = self.calculate_bbox_loss(pd_distri, pd_bboxes_pyramid, anc_points, target_scores,  target_bboxes, stride_tensor, target_scores_sum, TF_fg) #loss_iou, loss_dfl
            
            loss[0], loss[3] = self.bbox_loss(pd_distri, pd_bboxes_pyramid, anc_points, target_scores,target_bboxes / stride_tensor,
                                              target_scores_sum,TF_fg, ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1, batch_size).transpose(1, 0))

        # return self.bbox_loss(
        #     pd_distri, #torch.Size([2, 8400, 64])
        #     pd_bboxes_pyramid, #torch.Size([2, 8400, 4])
        #     anc_points, #torch.Size([8400, 2])
        #     target_scores, #torch.Size([2, 8400, 80])
        #     target_bboxes / stride_tensor, #torch.Size([2, 8400, 4])
            
        #     target_scores_sum,
        #     TF_fg,
        # )

            tgt_masks = self.prepare_masks(batch, mask_h, mask_w) #torch.Size([2, 160, 160])
            loss[1] = self.calculate_segmentation_loss(tgt_masks, #分割损失
                                                       target_bboxes,
                                                       
                                                       fg_inds,
                                                       TF_fg,
                                                       batch["img_idx"],
                                                       
                                                       pd_seg_matrix,
                                                       proto,

                                                       imgsz,
                                                       self.overlap_mask)
        else:
            loss[1] += (proto * 0).sum() + (pd_seg_matrix * 0).sum()
        
        loss = self.apply_gains(loss) #iou 分割 分类 dfl
        
        return loss.sum() * batch_size, loss.detach() #批次总损失  单次平均损失 
    
    def unpack_preds(self, preds):
        if len(preds) == 3:
            return preds
        else:
            return preds[1]

    def prepare_predictions(self, feats, pd_seg_matrix):
        pd_distri, pd_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split((self.reg_max * 4, self.nc), 1) #torch.Size([2, 64, 8400]) torch.Size([2, 80, 8400])
        pd_scores = pd_scores.permute(0, 2, 1).contiguous()
        pd_distri = pd_distri.permute(0, 2, 1).contiguous()
        pd_seg_matrix = pd_seg_matrix.permute(0, 2, 1).contiguous()
        return pd_distri, pd_scores, pd_seg_matrix



    def generate_targets(self, batch, imgsz, batch_size):
        try:

            targets = torch.cat((batch["img_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1) #torch.Size([21, 6]) 框 6
            gt_targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]) #计算机图像域框xyxy
            gt_labels, gt_bboxes = gt_targets.split((1, 4), 2)  #handled_targets是原始图像xyxy并且 批 框 维
            gt_mask = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
            return gt_labels, gt_bboxes, gt_mask #torch.Size([2, 13, 1]) torch.Size([2, 13, 4]) torch.Size([2, 13, 1])
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

    def prepare_masks(self, batch, mask_h, mask_w):
        masks = batch["masks"].to(self.device).float() #torch.Size([2, 160, 160])
        if tuple(masks.shape[-2:]) != (mask_h, mask_w):
            masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
        return masks

    def calculate_bbox_loss(self, pd_distri, pd_bboxes_pyramid, anc_points, target_scores,  target_bboxes, stride_tensor, target_scores_sum, TF_fg):
        return self.bbox_loss(
            pd_distri, #torch.Size([2, 8400, 64])
            pd_bboxes_pyramid, #torch.Size([2, 8400, 4])
            anc_points, #torch.Size([8400, 2])
            target_scores, #torch.Size([2, 8400, 80])
            target_bboxes / stride_tensor, #torch.Size([2, 8400, 4])
            
            target_scores_sum,
            TF_fg,
        )

    def apply_gains(self, loss):
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain
        return loss

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """ #40个实例都有掩膜 32维度信息
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        # loss = self.tversky_loss(pred_mask, gt_mask)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        # loss = combined_loss(pred_mask, gt_mask)
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()






    def calculate_segmentation_loss(
        self,
        tgt_masks: torch.Tensor, #torch.Size([2, 160, 160])
        target_bboxes: torch.Tensor, #torch.Size([2, 8400, 4])
        
        fg_inds: torch.Tensor, #torch.Size([2, 8400])
        TF_fg: torch.Tensor, #torch.Size([2, 8400])
        img_idx: torch.Tensor, #torch.Size([21])
        

        pd_seg_matrix: torch.Tensor, #torch.Size([2, 8400, 32])
        proto: torch.Tensor, #torch.Size([2, 32, 160, 160])
        
        imgsz: torch.Tensor, #torch.Size([2])
        overlap: bool,
    ) -> torch.Tensor:

        _, _, mask_h, mask_w = proto.shape #160 160
        loss = 0

        # Normalize to 0-1
        normalized_target_bboxes = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Normalize to mask size
        mxyxy = normalized_target_bboxes * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        # Areas of target bboxes
        marea = xyxy2xywh(normalized_target_bboxes)[..., 2:].prod(2) #-》torch.Size([2, 8400, 4]) -》torch.Size([2, 8400]) 整个目标框的面积   浅拷贝


        # torch.Size([160, 160]) torch.Size([8400])torch.Size([8400])torch.Size([8400, 32])torch.Size([32, 160, 160])torch.Size([8400, 4]) torch.Size([8400])
        for i, (gt_masks_i,fg_inds_i,TF_fg_i,pd_seg_matrix_i,proto_i,mxyxy_i,marea_i)in enumerate(
                zip(tgt_masks, fg_inds, TF_fg, pd_seg_matrix, proto, mxyxy, marea )):
            if TF_fg_i.any(): #如果是是哪个前景点  是不是前景点
                mask_idx = fg_inds_i[TF_fg_i] #torch.Size([8400])-》torch.Size([40])
                if overlap:
                    gt_class_mask = gt_masks_i == (mask_idx + 1).view(-1, 1, 1) #-》torch.Size([40, 160, 160])
                    gt_class_mask = gt_class_mask.float()
                else:
                    gt_class_mask = gt_masks_i[img_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_class_mask, pd_seg_matrix_i[TF_fg_i], proto_i, mxyxy_i[TF_fg_i], marea_i[TF_fg_i]
                ) #torch.Size([40, 160, 160]) torch.Size([40, 32]) torch.Size([32, 160, 160]) torch.Size([40, 4]) torch.Size([40, 4])

            # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pd_seg_matrix * 0).sum()  # inf sums may lead to nan loss


        return loss / TF_fg.sum()







class v8PoseLoss(DetectionLoss):
    """Criterion class for computing training avg_loss_items."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pd_distri, pd_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pd_scores = pd_scores.permute(0, 2, 1).contiguous()
        pd_distri = pd_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pd_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anc_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pd_scores.shape[0]
        img_idx = batch["img_idx"].view(-1, 1)
        targets = torch.cat((img_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        gt_mask = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pd_bboxes = self.bbox_decode(anc_points, pd_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anc_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, TF_fg, fg_inds = self.assigner(
            pd_scores.detach().sigmoid(),
            (pd_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anc_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            gt_mask,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pd_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pd_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if TF_fg.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pd_distri, pd_bboxes, anc_points, target_bboxes, target_scores, target_scores_sum, TF_fg
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                TF_fg, fg_inds, keypoints, img_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anc_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anc_points[:, [0]] - 0.5
        y[..., 1] += anc_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, fg_inds, keypoints, img_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            fg_inds (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            img_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        img_idx = img_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(img_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on img_idx
        for i in range(batch_size):
            keypoints_i = keypoints[img_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of fg_inds to match the shape of batched_keypoints
        target_gt_idx_expanded = fg_inds.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training avg_loss_items."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(DetectionLoss):
    def __init__(self, model):
        """
        Initializes v8OBBLoss with model, assigner, and rotated bbox loss.

        Note model must be de-paralleled.
        """
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, nc=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pd_distri, pd_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pd_scores = pd_scores.permute(0, 2, 1).contiguous()
        pd_distri = pd_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pd_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anc_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            img_idx = batch["img_idx"].view(-1, 1)
            targets = torch.cat((img_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            gt_mask = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pd_bboxes = self.bbox_decode(anc_points, pd_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pd_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, TF_fg, _ = self.assigner(
            pd_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anc_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            gt_mask,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pd_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pd_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if TF_fg.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pd_distri, pd_bboxes, anc_points, target_bboxes, target_scores, target_scores_sum, TF_fg
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anc_points, pd_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anc_points (torch.Tensor): Anchor points, (h*w, 2).
            pd_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pd_dist.shape  # batch, anchors, channels
            pd_dist = pd_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pd_dist.dtype))
        return torch.cat((dist2rbox(pd_dist, pred_angle, anc_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training avg_loss_items."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection avg_loss_items using the provided model."""
        self.one2many = DetectionLoss(model, tal_topk=10)
        self.one2one = DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
