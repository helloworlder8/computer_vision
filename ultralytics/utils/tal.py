# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from .checks import check_version
from .metrics import bbox_iou, probiou
from .ops import xywhr2xyxyxyxy

TORCH_1_10 = check_version(torch.__version__, "1.10.0")


class TaskAlignedAssigner(nn.Module):
    """
        任务对齐分配
    """

    def __init__(self, topk=10, nc=80, alpha=1.0, beta=6.0, eps=1e-9,IoU_algorithm="CIoU"):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk
        self.nc = nc
        self.bg_idx = nc
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.IoU_algorithm = IoU_algorithm

    @torch.no_grad()
    def forward(self, pd_cls, pd_bboxes, anc_points, gt_cls, gt_bboxes, gt_bool):
        """  
        torch.Size([2, 8400, 80])
        torch.Size([2, 8400, 4])
        torch.Size([8400, 2])
        
        torch.Size([2, 7, 1])
        torch.Size([2, 7, 2])
        torch.Size([2, 7, 1])
        """
        self.bs = pd_bboxes.shape[0] #2
        self.max_boxes = gt_bboxes.shape[1]  #7

        if self.max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_cls[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_cls).to(device),
                torch.zeros_like(pd_cls[..., 0]).to(device),
                torch.zeros_like(pd_cls[..., 0]).to(device),
            )

        match_topk, align_scores, iou_scores = self._generate_positive_sample(
            pd_cls, pd_bboxes, anc_points, gt_cls, gt_bboxes, gt_bool
        ) #-》torch.Size([2, 7, 8400]) torch.Size([2, 7, 8400]) torch.Size([2, 7, 8400])
        match_inds, match_bool, updated_match_topk = self.get_fg_mask_and_indices(match_topk, iou_scores, self.max_boxes)#-> torch.Size([2, 8400]) torch.Size([2, 8400]) torch.Size([2,7, 8400]) 站在anchor的角度

        # Assigned target
        expend_gt_cls, gt_bboxes_matrix, gt_cls_matrix = self.assign_targets(gt_cls, gt_bboxes, match_inds, match_bool)

        # Normalize
        align_scores *= updated_match_topk #torch.Size([2, 7, 8400])
        pos_align_metrics = align_scores.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (iou_scores * updated_match_topk).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_scores * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        gt_cls_matrix = gt_cls_matrix * norm_align_metric

        return expend_gt_cls, gt_bboxes_matrix, gt_cls_matrix, match_inds, match_bool.bool()
    # gt_bool
    def _generate_positive_sample(self, pd_cls, pd_bboxes, anc_points, gt_cls, gt_bboxes, gt_bool):
        # torch.Size([2, 8400, 80]) torch.Size([2, 8400, 4]) torch.Size([8400, 2]) torch.Size([2, 7, 1]) torch.Size([2, 7, 4]) torch.Size([2, 7, 1])
        anc_in_gt_bboxes = self._candidates_anc_in_gt_bboxes(anc_points, gt_bboxes) #缩小范围
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_scores, iou_scores = self._compute_align_iou_scores(pd_cls, pd_bboxes, gt_cls, gt_bboxes, anc_in_gt_bboxes * gt_bool) #得到两个值
        # torch.Size([2, 7, 8400])  torch.Size([2, 7, 8400])
        match_topk = self.select_topk_candidates(align_scores, topk_mask=gt_bool.expand(-1, -1, self.topk).bool()) #进一步缩小范围
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        match_topk = gt_bool * anc_in_gt_bboxes * match_topk #大截断 边界内 数字化前10个

        return match_topk, align_scores, iou_scores #计算损失的点 综合得分 iou得分


    def _compute_align_iou_scores(self, pd_cls, pd_bboxes, gt_cls, gt_bboxes, anc_in_gt_bboxes):
        """  torch.Size([2, 8400, 80])  torch.Size([2, 8400, 4]) torch.Size([2, 7, 1]) torch.Size([2, 7, 4]) torch.Size([2, 7, 8400])
        Compute alignment metric given predicted and ground truth bounding boxes.
        
        Args:
            pd_cls (torch.Tensor): Predicted scores of shape (bs, num_anchors, nc)
            pd_bboxes (torch.Tensor): Predicted bounding boxes of shape (bs, num_anchors, 4)
            gt_cls (torch.Tensor): Ground truth labels of shape (bs, max_boxes, 1)
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (bs, max_boxes, 4)
            anc_in_gt_bboxes (torch.Tensor): Mask for ground truth presence of shape (bs, max_boxes, num_anchors)

        Returns:
            torch.Tensor: Alignment metric
            torch.Tensor: Overlaps between predicted and ground truth boxes
        """ #gt_bool
        num_anc = pd_bboxes.shape[-2]  # 8400
        anc_in_gt_bboxes = anc_in_gt_bboxes.bool()  # torch.Size([2, 7, 8400])

        # Initialize tensors for iou_scores and predicted scores
        iou_scores = torch.zeros((self.bs, self.max_boxes, num_anc), dtype=pd_bboxes.dtype, device=pd_bboxes.device) #torch.Size([2, 7, 8400]) 批 框 维
        cls_scores = torch.zeros((self.bs, self.max_boxes, num_anc), dtype=pd_cls.dtype, device=pd_cls.device) #torch.Size([2, 7, 8400])

        # Prepare indices for selecting scores
        img_labels_ind = torch.zeros((2, self.bs, self.max_boxes), dtype=torch.long) #torch.Size([2, 2, 7])
        img_labels_ind[0] = torch.arange(self.bs).view(-1, 1).expand(-1, self.max_boxes)  #torch.Size([2, 7]) 批索引
        img_labels_ind[1] = gt_cls.squeeze(-1)  # Class indices #torch.Size([2, 7]) 类索引

        # Select predicted scores using batch and class indices #预测包括那一批 哪一个点 哪一个类
        cls_scores[anc_in_gt_bboxes] = pd_cls[img_labels_ind[0], :, img_labels_ind[1]][anc_in_gt_bboxes]
        # torch.Size([2, 8400, 80]) torch.Size([2, 7]) torch.Size([2, 7, 8400])->torch.Size([2, 7, 8400])
        # Expand and mask predicted and ground truth boxes for IoU calculation
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.max_boxes, -1, -1)[anc_in_gt_bboxes] #torch.Size([2, 8400, 4])->torch.Size([3328, 4])
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, num_anc, -1)[anc_in_gt_bboxes]

        # Calculate iou_scores using IoU calculation function
        iou_scores[anc_in_gt_bboxes] = self.iou_calculation(gt_boxes, pd_boxes) #-》torch.Size([2, 7, 8400])

        # Calculate alignment metric
        align_scores = cls_scores.pow(self.alpha) * iou_scores.pow(self.beta)

        return align_scores, iou_scores

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for horizontal bounding boxes."""
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
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, **IoU_options).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, align_scores, largest=True, topk_mask=None):
        # Select top-k metrics and their indices
        topk_metrics, topk_idxs = torch.topk(align_scores, self.topk, dim=-1, largest=largest) #torch.Size([2, 7, 10]) torch.Size([2, 7, 10])

        # If topk_mask is not provided, compute it based on the topk_metrics
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)

        # Fill invalid indices with zeros
        topk_idxs.masked_fill_(~topk_mask, 0) #torch.Size([2, 7, 10])

        # Initialize count tensor for counting top-k occurrences
        match_topk = torch.zeros(align_scores.shape, dtype=torch.int8, device=topk_idxs.device) #torch.Size([2, 7, 8400])
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device) #torch.Size([2, 7, 1])

        # Scatter add ones to match_topk at positions specified by topk_idxs
        for k in range(self.topk):
            match_topk.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)

        # Mask out counts greater than 1
        match_topk.masked_fill_(match_topk > 1, 0)

        return match_topk.to(align_scores.dtype)

    def assign_targets(self, gt_cls, gt_bboxes, match_inds, match_bool):
        """
        Assigns target labels, bounding boxes, and scores to the predicted boxes based on the selected indices.

        Args:
            match_inds (Tensor): Indices of the selected boxes, shape (b, h*w).
            gt_cls (Tensor): Ground truth labels, shape (b, max_num_obj, 1).
            gt_bboxes (Tensor): Ground truth bounding boxes, shape (b, max_num_obj, 4).
            match_bool (Tensor): Mask indicating foreground boxes, shape (b, h*w).

        Returns:
            expend_gt_cls (Tensor): Assigned target labels, shape (b, h*w).
            gt_bboxes_matrix, (Tensor): Assigned target bounding boxes, shape (b, h*w, 4).
            gt_cls_matrix (Tensor): One-hot encoded target scores, shape (b, h*w, nc).
        """
        # Compute batch indices and update selected box indices
        batch_inds = torch.arange(self.bs, dtype=torch.int64, device=gt_cls.device)[..., None]  # torch.Size([2, 1])
        match_inds = match_inds + batch_inds * self.max_boxes  #魔鬼细节 (b, h*w) torch.Size([2, 8400])

        # Retrieve target labels
        expend_gt_cls = gt_cls.long().flatten()[match_inds]  # (b, h*w)-》 torch.Size([2, 8400])
        expend_gt_cls.clamp_(0)  # Ensure no negative labels

        # Retrieve target bounding boxes
        gt_bboxes_matrix = gt_bboxes.view(-1, gt_bboxes.shape[-1])[match_inds]  # (b, h*w, 4) torch.Size([2, 8400, 4])

        # Initialize target scores
        gt_cls_matrix = torch.zeros(
            (expend_gt_cls.shape[0], expend_gt_cls.shape[1], self.nc),
            dtype=torch.int64,
            device=expend_gt_cls.device,
        )  # (b, h*w, nc) torch.Size([2, 8400, 80])

        # Assign one-hot encoded target scores
        gt_cls_matrix.scatter_(2, expend_gt_cls.unsqueeze(-1), 1)  #torch.Size([2, 8400, 80])

        # Apply foreground mask to target scores
        fg_scores_mask = match_bool[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, nc)
        gt_cls_matrix = torch.where(fg_scores_mask > 0, gt_cls_matrix, 0) #最后一个维度统一

        return expend_gt_cls, gt_bboxes_matrix, gt_cls_matrix #


    @staticmethod
    def _candidates_anc_in_gt_bboxes(anc_points, gt_bboxes, eps=1e-9):
        """          #torch.Size([8400, 2]) torch.Size([2, 7, 4])
         Select positive anchor centers within ground truth bounding boxes.

        Args:
            anc_points (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, num_bboxes, 4)

        Returns:
            (Tensor): shape(b, num_bboxes, h*w)
        """
        num_anc = anc_points.shape[0] #点 维 8400
        bs, num_bboxes, _ = gt_bboxes.shape # 2 7
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # torch.Size([14, 1, 2])
        bbox_deltas = torch.cat((anc_points[None] - lt, rb - anc_points[None]), dim=2).view(bs, num_bboxes, num_anc, -1) #每一个框 每一个点 偏移
        # torch.Size([2, 7, 8400, 4])
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def get_fg_mask_and_indices(match_topk, iou_scores, max_num_boxes):
        """
        If an anchor box is assigned to multiple ground truths (gts), select the one with the highest IoU.

        Args:
            match_topk (Tensor): shape (bs, max_num_boxes, num_anchors), binary mask indicating assigned boxes.
            iou_scores (Tensor): shape (bs, max_num_boxes, num_anchors), IoU scores between predicted and gt boxes.
            max_num_boxes (int): maximum number of boxes per image.

        Returns:
            match_inds (Tensor): shape (bs, num_anchors), indices of the selected boxes for each anchor.
            match_bool (Tensor): shape (bs, num_anchors), binary mask indicating foreground anchors.
            updated_match_topk (Tensor): shape (bs, max_num_boxes, num_anchors), updated mask with highest IoU selections.
        """
        # Calculate the number of foreground anchors for each anchor (bs, num_anchors)
        match_bool = match_topk.sum(-2)

        if match_bool.max() > 1:  # If an anchor is assigned to multiple gts
            # Create a mask for anchors assigned to multiple gts (bs, max_num_boxes, num_anchors)
            multiple_gt_mask = match_bool.unsqueeze(1).expand(-1, max_num_boxes, -1) > 1

            # Find the gt with the highest IoU for each anchor (bs, num_anchors)
            highest_iou_indices = iou_scores.argmax(1)

            # Create a tensor to mark the highest IoU overlaps (bs, max_num_boxes, num_anchors)
            highest_iou_mask = torch.zeros_like(match_topk)
            highest_iou_mask.scatter_(1, highest_iou_indices.unsqueeze(1), 1)

            # Update match_topk to keep only the highest IoU overlaps for anchors assigned to multiple gts
            updated_match_topk = torch.where(multiple_gt_mask, highest_iou_mask, match_topk).float()
            
            # Recalculate match_bool after updating match_topk
            match_bool = updated_match_topk.sum(-2)
        else:
            updated_match_topk = match_topk

        # Find the index of the gt that each anchor serves (bs, num_anchors)
        match_inds = updated_match_topk.argmax(-2) #torch.Size([2, 7, 8400]) torch.Size([2, 8400])

        return match_inds, match_bool, updated_match_topk



class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def _candidates_anc_in_gt_bboxes(anc_points, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            anc_points (Tensor): shape(h*w, 2)
            gt_bboxes (Tensor): shape(b, num_bboxes, 5)

        Returns:
            (Tensor): shape(b, num_bboxes, h*w)
        """
        # (b, num_bboxes, 5) --> (b, num_bboxes, 4, 2)
        corners = xywhr2xyxyxyxy(gt_bboxes)
        # (b, num_bboxes, 1, 2)
        a, b, _, d = corners.split(1, dim=-2)
        ab = b - a
        ad = d - a

        # (b, num_bboxes, h*w, 2)
        ap = anc_points - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(feats, strides, grid_cell_offset=0.5): #[torch.Size([2, 144, 80, 80]), torch.Size([2, 144, 40, 40]), torch.Size([2, 144, 20, 20])]  torch.Size([3])
    """Generate anchors from features.""" #竟是用些细枝末节的东西
    anc_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape  #torch.Size([2, 144, 32, 84])
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx) #torch.Size([80, 80])
        anc_points.append(torch.stack((sx, sy), -1).view(-1, 2)) #[torch.Size([6400, 2])]
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device)) #[torch.Size([6400, 1])]
    return torch.cat(anc_points), torch.cat(stride_tensor)


def dist2bbox(dist, anc_points, xywh=True, dim=-1): #预测的是每一个点的偏移量
    """Transform dist(ltrb) to box(xywh or xyxy)."""
    lt, rb = dist.chunk(2, dim) #torch.Size([2, 2, 8400]) torch.Size([2, 2, 8400])
    x1y1 = anc_points - lt
    x2y2 = anc_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2 #中心点
        wh = x2y2 - x1y1         #宽高
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anc_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anc_points - x1y1, x2y2 - anc_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)


def dist2rbox(pred_dist, pred_angle, anc_points, dim=-1):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).
        anc_points (torch.Tensor): Anchor points, (h*w, 2).
    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, (bs, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (bs, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anc_points
    return torch.cat([xy, lt + rb], dim=dim)
