# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from .metrics import bbox_iou, probiou
from .tal import bbox2dist


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
        target_class_mask = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * target_class_mask)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with num_outchannel parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()

# 回归损失
class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
                # anchor_points, pd_codebboxes_dist, pd_bboxes, target_bboxes_BA4matrix, target_class_BACmatrix, target_class_scores, mask_poss_sum_BAmatrix
    def forward(self, anchor_points, pd_codebboxes_dist, pd_bboxes, target_bboxes_BA4matrix, target_class_BACmatrix, target_class_scores, mask_poss_sum_BAmatrix):
        """IoU loss."""#8400 2        2 8400 64          2 8400 4     2 8400 4              2 8400 1                   29值                           2 8400
        target_class_mask = target_class_BACmatrix.sum(-1)[mask_poss_sum_BAmatrix].unsqueeze(-1) #torch.Size([90,1]) 认为每一个点的权重是不一样的
        iou = bbox_iou(pd_bboxes[mask_poss_sum_BAmatrix], target_bboxes_BA4matrix[mask_poss_sum_BAmatrix], xywh=False, CIoU=True) #x1 y1 x2 y2用的两顶点坐标
        # iou = bbox_iou(pd_bboxes[mask_poss_sum_BAmatrix], target_bboxes_BA4matrix[mask_poss_sum_BAmatrix], xywh=False, SIoU=True)

        loss_iou = ((1.0 - iou) * target_class_mask).sum() / target_class_scores  #分管权重 总得分

        # DFL loss
        if self.use_dfl:
            target_dist = bbox2dist(anchor_points, target_bboxes_BA4matrix, self.reg_max) #torch.Size([8400, 2]) torch.Size([2, 8400, 4])-》torch.Size([2, 8400, 4])
            loss_dfl = self._df_loss(pd_codebboxes_dist[mask_poss_sum_BAmatrix].view(-1, self.reg_max + 1), target_dist[mask_poss_sum_BAmatrix]) * target_class_mask 
            loss_dfl = loss_dfl.sum() / target_class_scores #torch.Size([2, 8400, 64]) torch.Size([360, 16])
        else:
            loss_dfl = torch.tensor(0.0).to(pd_codebboxes_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pd_codebboxes_dist, target_dist): #torch.Size([360, 16]) #torch.Size([90, 4])
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391  确定每个框中的每一个点的位置
        """
        tl = target_dist.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target_dist  # target_class_mask left
        wr = 1 - wl  # target_class_mask right
        return (
            F.cross_entropy(pd_codebboxes_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pd_codebboxes_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_dfl)

    def forward(self, pd_codebboxes_dist, pd_bboxes, anchor_points, target_bboxes_BA4matrix, target_class_BACmatrix, target_class_scores, mask_poss_sum_BAmatrix):
        """IoU loss."""
        target_class_mask = target_class_BACmatrix.sum(-1)[mask_poss_sum_BAmatrix].unsqueeze(-1)
        iou = probiou(pd_bboxes[mask_poss_sum_BAmatrix], target_bboxes_BA4matrix[mask_poss_sum_BAmatrix])
        loss_iou = ((1.0 - iou) * target_class_mask).sum() / target_class_scores

        # DFL loss
        if self.use_dfl:
            target_dist = bbox2dist(anchor_points, xywh2xyxy(target_bboxes_BA4matrix[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pd_codebboxes_dist[mask_poss_sum_BAmatrix].view(-1, self.reg_max + 1), target_dist[mask_poss_sum_BAmatrix]) * target_class_mask
            loss_dfl = loss_dfl.sum() / target_class_scores
        else:
            loss_dfl = torch.tensor(0.0).to(pd_codebboxes_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas).pow(2) / (area + 1e-9) / 2  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class Anchor_Free_Detection_Loss: #构造函数的时候把整个模型传进来了
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes Anchor_Free_Detection_Loss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        hyp = model.args  # hyperparameters

        m = model.seqential_model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = hyp
        self.strides_list = m.stride  # model strides
        self.num_class = m.num_class  # number of classes
        self.num_outchannel = m.num_outchannel
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_class=self.num_class, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocessing_targets(self, targets, batch_size, imgsz): #torch.Size([14, 6])  2   640 640 640 640
        """Preprocesses the target targets_img_count and TF with the input batch_labels_list size to output a tensor."""
        if targets.shape[0] == 0:
            preproc_targets = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            targets_img_ind = targets[:, 0]  # image index
            _, targets_img_count = targets_img_ind.unique(return_counts=True) #标签对应图片索引
            targets_img_count = targets_img_count.to(dtype=torch.int32)  #8 6
            preproc_targets = torch.zeros(batch_size, targets_img_count.max(), 5, device=self.device)#285 两张图片 每张图片最多8个目标 5
            for j in range(batch_size):
                TF = targets_img_ind == j
                num_img_corres_tag = TF.sum()#这一批图像标签为某一个值的数目 8
                if num_img_corres_tag:
                    preproc_targets[j, :num_img_corres_tag] = targets[TF, 1:] # torch.Size([8, 5]) 筛选
            preproc_targets[..., 1:5] = xywh2xyxy(preproc_targets[..., 1:5].mul_(imgsz)) #bbox 操作 输入xywh输出xyxy
        return preproc_targets  #只有class bbox

    def bbox_decode(self, anchor_points, pd_codebboxes_dist):#torch.Size([8400, 2]) torch.Size([2, 8400, 64])
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pd_codebboxes_dist.shape  # batch_size, anchors, channels
            pd_codebboxes_dist = pd_codebboxes_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pd_codebboxes_dist.dtype)) #->torch.Size([2, 8400, 4])
        return dist2bbox(pd_codebboxes_dist, anchor_points, xywh=False) #torch.Size([2, 8400, 4]) torch.Size([8400, 2])
        # pd_codebboxes预测的是偏差  偏差转换成bbox


    def _process_predictions(self, preds):
        """ 预测值处理 """

        pd_codebboxes_dist, pd_class = torch.cat([xi.view(preds[0].shape[0], self.num_outchannel, -1) for xi in preds], 2).split(
            (self.reg_max * 4, self.num_class), 1
        ) #torch.Size([2, 64, 8400])  #torch.Size([2, 80, 8400])   宽高压缩  pd_codebboxes这个是网络直接输出的
        pd_codebboxes_dist = pd_codebboxes_dist.permute(0, 2, 1).contiguous() #torch.Size([2, 8400, 64])
        pd_class = pd_class.permute(0, 2, 1).contiguous() #torch.Size([2, 8400, 80])
        return pd_codebboxes_dist, pd_class


    def _process_targets(self, batch_labels_list,batch_size,imgsz):
        targets = torch.cat((batch_labels_list["batch_idx"].view(-1, 1), batch_labels_list["cls"].view(-1, 1), batch_labels_list["bboxes"]), 1)#torch.Size([14, 6])
        preproc_targets = self.preprocessing_targets(targets.to(self.device), batch_size, imgsz=imgsz[[1, 0, 1, 0]]) #285  输出xyxy
        gt_class, gt_bboxes = preproc_targets.split((1, 4), 2)  # cls, xyxy  torch.Size([2, 8, 5])
        return gt_class, gt_bboxes

    # 计算损失值  训练的preds和验证的preds不一样
    def __call__(self, preds, batch_labels_list): #torch.Size([2, 144, 80, 80]) #torch.Size([2, 144, 40, 40]) torch.Size([2, 144, 20, 20])
        """
        计算基于batch_labels_list大小的框、类别和dfl损失之和。
        输入预测维度示例: torch.Size([2, 144, 80, 80]), torch.Size([2, 144, 40, 40]), torch.Size([2, 144, 20, 20])
        """
        
        """ 定义三个损失 """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        preds = preds[1] if isinstance(preds, tuple) else preds
        dtype = preds[0].dtype
        batch_size = preds[0].shape[0]
        imgsz = torch.tensor(preds[0].shape[2:], device=self.device, dtype=dtype) * self.strides_list[0] #标准图像尺寸



        """ anchor处理 """
        anchor_points, stride_points = make_anchors(preds, self.strides_list, 0.5) #-> 8400 2   8400 1


        """ 预测处理 """
        # ([2, 8400, 64]) ([2, 8400, 80])
        pd_codebboxes_dist, pd_class =  self._process_predictions(preds) #先要处理预测值 偏差  预测值就是编码了的并且是分布的
        pd_bboxes = self.bbox_decode(anchor_points, pd_codebboxes_dist)  #softmax xyxy  ->torch.Size([2, 8400, 4]) 实际值



        """ 目标值处理 """
        gt_class, gt_bboxes =  self._process_targets(batch_labels_list,batch_size,imgsz) 
        mask_gt_BT1 = gt_bboxes.sum(2, keepdim=True).gt_(0) 

        """ 编码成2 8400 ？的形式 """
        # return target_bboxes_BA4matrix, target_bboxes_BA4matrix, target_class_BACmatrix, mask_poss_sum_BAmetrics.bool(), mask_poss_idx_BAmetrics
        _, target_bboxes_BA4matrix, target_class_BACmatrix, mask_poss_sum_BAmatrix, _ = self.assigner.forward(
            anchor_points * stride_points, #  8400 2

            pd_class.detach().sigmoid(), #torch.Size([2, 8400, 80])   sigmoid
            (pd_bboxes.detach() * stride_points).type(gt_bboxes.dtype), #torch.Size([2, 8400, 4])

            gt_class, #torch.Size([2, 8, 1])
            gt_bboxes, #torch.Size([2, 8, 4])
            mask_gt_BT1,   #torch.Size([2, 8, 1])
        )

        target_class_scores = max(target_class_BACmatrix.sum(), 1) 

        # Cls loss
        # loss[1] = self.varifocal_loss(pd_class, target_class_BACmatrix, target_labels_dict) / target_class_scores  # VFL way
        #                  2 8400 80
        loss[1] = self.bce(pd_class, target_class_BACmatrix.to(dtype)).sum() / target_class_scores  # 分类损失

        # Bbox loss
        if mask_poss_sum_BAmatrix.sum():
            target_bboxes_BA4matrix /= stride_points
            loss[0], loss[2] = self.bbox_loss(      #def forward(self, pd_codebboxes_di
                anchor_points, pd_codebboxes_dist, pd_bboxes, target_bboxes_BA4matrix, target_class_BACmatrix, target_class_scores, mask_poss_sum_BAmatrix
            )  #8400 2             64            8400 4         2 8400 4                2 8400 80                    29             2 8400

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(Anchor_Free_Detection_Loss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch_labels_list):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        preds, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch_labels_list size, number of masks, mask height, mask width
        pd_codebboxes_dist, pd_class = torch.cat([xi.view(preds[0].shape[0], self.num_outchannel, -1) for xi in preds], 2).split(
            (self.reg_max * 4, self.num_class), 1
        )

        # B, grids, ..
        pd_class = pd_class.permute(0, 2, 1).contiguous()
        pd_codebboxes_dist = pd_codebboxes_dist.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pd_class.dtype
        imgsz = torch.tensor(preds[0].shape[2:], device=self.device, dtype=dtype) * self.strides_list[0]  # image size (h,w)
        anchor_points, stride_points = make_anchors(preds, self.strides_list, 0.5)

        # Targets
        try:
            batch_idx = batch_labels_list["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch_labels_list["cls"].view(-1, 1), batch_labels_list["bboxes"]), 1)
            targets = self.preprocessing_targets(targets.to(self.device), batch_size, imgsz=imgsz[[1, 0, 1, 0]])
            gt_class, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt_BT1 = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pd_bboxes = self.bbox_decode(anchor_points, pd_codebboxes_dist)  # xyxy, (b, h*w, 4)

        _, target_bboxes_BA4matrix, target_class_BACmatrix, mask_poss_sum_BAmatrix, target_gt_idx = self.assigner(
            pd_class.detach().sigmoid(),
            (pd_bboxes.detach() * stride_points).type(gt_bboxes.dtype),
            anchor_points * stride_points,
            gt_class,
            gt_bboxes,
            mask_gt_BT1,
        )

        target_class_scores = max(target_class_BACmatrix.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pd_class, target_class_BACmatrix, target_labels_dict) / target_class_scores  # VFL way
        loss[2] = self.bce(pd_class, target_class_BACmatrix.to(dtype)).sum() / target_class_scores  # BCE

        if mask_poss_sum_BAmatrix.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(#mark
                pd_codebboxes_dist,
                pd_bboxes,
                anchor_points,
                target_bboxes_BA4matrix / stride_points,
                target_class_BACmatrix,
                target_class_scores,
                mask_poss_sum_BAmatrix,
            )
            # Masks loss
            masks = batch_labels_list["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                mask_poss_sum_BAmatrix, masks, target_gt_idx, target_bboxes_BA4matrix, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

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
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        mask_poss_sum_BAmatrix: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes_BA4matrix: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            mask_poss_sum_BAmatrix (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes_BA4matrix (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch_labels_list loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes_BA4matrix / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(mask_poss_sum_BAmatrix, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / mask_poss_sum_BAmatrix.sum()


class v8PoseLoss(Anchor_Free_Detection_Loss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch_labels_list):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        preds, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pd_codebboxes_dist, pd_class = torch.cat([xi.view(preds[0].shape[0], self.num_outchannel, -1) for xi in preds], 2).split(
            (self.reg_max * 4, self.num_class), 1
        )

        # B, grids, ..
        pd_class = pd_class.permute(0, 2, 1).contiguous()
        pd_codebboxes_dist = pd_codebboxes_dist.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pd_class.dtype
        imgsz = torch.tensor(preds[0].shape[2:], device=self.device, dtype=dtype) * self.strides_list[0]  # image size (h,w)
        anchor_points, stride_points = make_anchors(preds, self.strides_list, 0.5)

        # Targets
        batch_size = pd_class.shape[0]
        batch_idx = batch_labels_list["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch_labels_list["cls"].view(-1, 1), batch_labels_list["bboxes"]), 1)
        targets = self.preprocessing_targets(targets.to(self.device), batch_size, imgsz=imgsz[[1, 0, 1, 0]])
        gt_class, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt_BT1 = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pd_bboxes = self.bbox_decode(anchor_points, pd_codebboxes_dist)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes_BA4matrix, target_class_BACmatrix, mask_poss_sum_BAmatrix, target_gt_idx = self.assigner(
            pd_class.detach().sigmoid(),
            (pd_bboxes.detach() * stride_points).type(gt_bboxes.dtype),
            anchor_points * stride_points,
            gt_class,
            gt_bboxes,
            mask_gt_BT1,
        )

        target_class_scores = max(target_class_BACmatrix.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pd_class, target_class_BACmatrix, target_labels_dict) / target_class_scores  # VFL way
        loss[3] = self.bce(pd_class, target_class_BACmatrix.to(dtype)).sum() / target_class_scores  # BCE

        # Bbox loss
        if mask_poss_sum_BAmatrix.sum():
            target_bboxes_BA4matrix /= stride_points
            loss[0], loss[4] = self.bbox_loss(#mark
                pd_codebboxes_dist, pd_bboxes, anchor_points, target_bboxes_BA4matrix, target_class_BACmatrix, target_class_scores, mask_poss_sum_BAmatrix
            )
            keypoints = batch_labels_list["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                mask_poss_sum_BAmatrix, target_gt_idx, keypoints, batch_idx, stride_points, target_bboxes_BA4matrix, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_points, target_bboxes_BA4matrix, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch_labels_list. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_points (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes_BA4matrix (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_points.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes_BA4matrix[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch_labels_list):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch_labels_list["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(Anchor_Free_Detection_Loss):
    def __init__(self, model):
        """
        Initializes v8OBBLoss with model, assigner, and rotated bbox loss.

        Note model must be de-paralleled.
        """
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.num_class, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max - 1, use_dfl=self.use_dfl).to(self.device)

    def preprocessing_targets(self, targets, batch_size, imgsz):
        """Preprocesses the target targets_img_count and TF with the input batch_labels_list size to output a tensor."""
        if targets.shape[0] == 0:
            preproc_targets = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, targets_img_count = i.unique(return_counts=True)
            targets_img_count = targets_img_count.to(dtype=torch.int32)
            preproc_targets = torch.zeros(batch_size, targets_img_count.max(), 6, device=self.device)
            for j in range(batch_size):
                TF = i == j
                num_img_corres_tag = TF.sum()
                if num_img_corres_tag:
                    bboxes = targets[TF, 2:]
                    bboxes[..., :4].mul_(imgsz)
                    preproc_targets[j, :num_img_corres_tag] = torch.cat([targets[TF, 1:2], bboxes], dim=-1)
        return preproc_targets

    def __call__(self, preds, batch_labels_list):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        preds, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch_labels_list size, number of masks, mask height, mask width
        pd_codebboxes_dist, pd_class = torch.cat([xi.view(preds[0].shape[0], self.num_outchannel, -1) for xi in preds], 2).split(
            (self.reg_max * 4, self.num_class), 1
        )

        # b, grids, ..
        pd_class = pd_class.permute(0, 2, 1).contiguous()
        pd_codebboxes_dist = pd_codebboxes_dist.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pd_class.dtype
        imgsz = torch.tensor(preds[0].shape[2:], device=self.device, dtype=dtype) * self.strides_list[0]  # image size (h,w)
        anchor_points, stride_points = make_anchors(preds, self.strides_list, 0.5)

        # targets
        try:
            batch_idx = batch_labels_list["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch_labels_list["cls"].view(-1, 1), batch_labels_list["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocessing_targets(targets.to(self.device), batch_size, imgsz=imgsz[[1, 0, 1, 0]])
            gt_class, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt_BT1 = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pd_bboxes = self.bbox_decode(anchor_points, pd_codebboxes_dist, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pd_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_points
        _, target_bboxes_BA4matrix, target_class_BACmatrix, mask_poss_sum_BAmatrix, _ = self.assigner(
            pd_class.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_points,
            gt_class,
            gt_bboxes,
            mask_gt_BT1,
        )

        target_class_scores = max(target_class_BACmatrix.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pd_class, target_class_BACmatrix, target_labels_dict) / target_class_scores  # VFL way
        loss[1] = self.bce(pd_class, target_class_BACmatrix.to(dtype)).sum() / target_class_scores  # BCE

        # Bbox loss
        if mask_poss_sum_BAmatrix.sum():
            target_bboxes_BA4matrix[..., :4] /= stride_points
            loss[0], loss[2] = self.bbox_loss(#mark
                pd_codebboxes_dist, pd_bboxes, anchor_points, target_bboxes_BA4matrix, target_class_BACmatrix, target_class_scores, mask_poss_sum_BAmatrix
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pd_codebboxes_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pd_codebboxes_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pd_codebboxes_dist.shape  # batch_labels_list, anchors, channels
            pd_codebboxes_dist = pd_codebboxes_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pd_codebboxes_dist.dtype))
        return torch.cat((dist2rbox(pd_codebboxes_dist, pred_angle, anchor_points), pred_angle), dim=-1)
