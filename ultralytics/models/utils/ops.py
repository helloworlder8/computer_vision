# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


class HungarianMatcher(nn.Module):
    """
    A module implementing the HungarianMatcher, which is a differentiable module to solve the assignment problem in an
    end-to-end fashion.

    HungarianMatcher performs optimal assignment over the predicted and ground truth bounding boxes using a cost
    function that considers classification scores, bounding box coordinates, and optionally, mask predictions.

    Attributes:
        cost_gain (dict): Dictionary of cost coefficients: 'class', 'bbox', 'giou', 'mask', and 'dice'.
        use_fl (bool): Indicates whether to use Focal Loss for the classification cost calculation.
        with_mask (bool): Indicates whether the model makes mask predictions.
        num_sample_points (int): The number of sample points used in mask cost calculation.
        alpha (float): The alpha factor in Focal Loss calculation.
        gamma (float): The gamma factor in Focal Loss calculation.

    Methods:
        forward(pd_bboxes, pd_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None): Computes the
            assignment between predictions and ground truths for a batch.
        _cost_mask(bs, num_gts, masks=None, gt_mask=None): Computes the mask cost and dice cost if masks are predicted.
    """

    def __init__(self, cost_gain=None, use_fl=True, with_mask=False, num_sample_points=12544, alpha=0.25, gamma=2.0):
        """Initializes HungarianMatcher with cost coefficients, Focal Loss, mask prediction, sample points, and alpha
        gamma factors.
        """
        super().__init__()
        if cost_gain is None:
            cost_gain = {"class": 1, "bbox": 5, "giou": 2, "mask": 1, "dice": 1}
        self.cost_gain = cost_gain
        self.use_fl = use_fl
        self.with_mask = with_mask
        self.num_sample_points = num_sample_points
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pd_bboxes, pd_scores, gt_bboxes, gt_cls, gt_groups, masks=None, gt_mask=None):
        """
        Forward pass for HungarianMatcher. This function computes costs based on prediction and ground truth
        (classification cost, L1 cost between boxes and GIoU cost between boxes) and finds the optimal matching between
        predictions and ground truth based on these costs.

        Args:
            pd_bboxes (Tensor): Predicted bounding boxes with shape [batch_size, num_queries, 4].
            pd_scores (Tensor): Predicted scores with shape [batch_size, num_queries, nc].
            gt_cls (torch.Tensor): Ground truth classes with shape [num_gts, ].
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape [num_gts, 4].
            gt_groups (List[int]): List of length equal to batch size, containing the number of ground truths for
                each image.
            masks (Tensor, optional): Predicted masks with shape [batch_size, num_queries, height, width].
                Defaults to None.
            gt_mask (List[Tensor], optional): List of ground truth masks, each with shape [num_masks, Height, Width].
                Defaults to None.

        Returns:
            (List[Tuple[Tensor, Tensor]]): A list of size batch_size, each element is a tuple (index_i, index_j), where:
                - index_i is the tensor of indices of the selected predictions (in order)
                - index_j is the tensor of indices of the corresponding selected ground truth targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_top_k, nc = pd_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, nc]
        pd_scores = pd_scores.detach().view(-1, nc) #torch.Size([1200, 80])
        pd_scores = F.sigmoid(pd_scores) if self.use_fl else F.softmax(pd_scores, dim=-1) #torch.Size([1200, 80])
        # [batch_size * num_queries, 4]
        pd_bboxes = pd_bboxes.detach().view(-1, 4) #torch.Size([1200, 4])

        # Compute the classification cost
        pd_scores = pd_scores[:, gt_cls] #torch.Size([1200, 30]) #缩小范围
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pd_scores**self.gamma) * (-(1 - pd_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pd_scores) ** self.gamma) * (-(pd_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pd_scores

        # Compute the L1 cost between boxes
        cost_bbox = (pd_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)  # (bs*num_queries, num_gt) #torch.Size([1200, 30, 4])->torch.Size([1200, 30])

        # Compute the GIoU cost between boxes, (bs*num_queries, num_gt)
        cost_giou = 1.0 - bbox_iou(pd_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1) #torch.Size([1200, 30])

        # Final cost matrix
        C = ( #torch.Size([1200, 30])
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox #l1损失
            + self.cost_gain["giou"] * cost_giou #giou损失
        ) 
        # Compute the mask cost and dice cost
        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        # Set invalid values (NaNs and infinities) to 0 (fixes ValueError: matrix contains invalid numeric entries)
        C[C.isnan() | C.isinf()] = 0.0

        C = C.view(bs, num_top_k, -1).cpu() #torch.Size([4, 300, 30]) 批 topk 框
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))] #匈牙利匹配 返回预测框和真实框匹配索引
        gt_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)  # (idx for queries, idx for gt)
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups[k])
            for k, (i, j) in enumerate(indices)
        ]

    # This function is for future RT-DETR Segment models
    # def _cost_mask(self, bs, num_gts, masks=None, gt_mask=None):
    #     assert masks is not None and gt_mask is not None, 'Make sure the input has `mask` and `gt_mask`'
    #     # all masks share the same set of points for efficient matching
    #     sample_points = torch.rand([bs, 1, self.num_sample_points, 2])
    #     sample_points = 2.0 * sample_points - 1.0
    #
    #     out_mask = F.grid_sample(masks.detach(), sample_points, align_corners=False).squeeze(-2)
    #     out_mask = out_mask.flatten(0, 1)
    #
    #     tgt_mask = torch.cat(gt_mask).unsqueeze(1)
    #     sample_points = torch.cat([a.repeat(b, 1, 1, 1) for a, b in zip(sample_points, num_gts) if b > 0])
    #     tgt_mask = F.grid_sample(tgt_mask, sample_points, align_corners=False).squeeze([1, 2])
    #
    #     with torch.cuda.amp.autocast(False):
    #         # binary cross entropy cost
    #         pos_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.ones_like(out_mask), reduction='none')
    #         neg_cost_mask = F.binary_cross_entropy_with_logits(out_mask, torch.zeros_like(out_mask), reduction='none')
    #         cost_mask = torch.matmul(pos_cost_mask, tgt_mask.T) + torch.matmul(neg_cost_mask, 1 - tgt_mask.T)
    #         cost_mask /= self.num_sample_points
    #
    #         # dice cost
    #         out_mask = F.sigmoid(out_mask)
    #         numerator = 2 * torch.matmul(out_mask, tgt_mask.T)
    #         denominator = out_mask.sum(-1, keepdim=True) + tgt_mask.sum(-1).unsqueeze(0)
    #         cost_dice = 1 - (numerator + 1) / (denominator + 1)
    #
    #         C = self.cost_gain['mask'] * cost_mask + self.cost_gain['dice'] * cost_dice
    #     return C

def get_cdn_group(targets, nc, num_queries, class_embedding, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0, training=False):
    if not training or num_dn <= 0:
        return None, None, None, None
    
    gt_groups = targets["gt_groups"] #[10, 10, 10]
    total_num = sum(gt_groups) #30
    batch_max_box = max(gt_groups) #10
    batch_size = len(gt_groups) #3
    if batch_max_box == 0:
        return None, None, None, None

    dn_eve_box = max(1, num_dn // batch_max_box) #10

    
    gt_cls = targets["cls"] #torch.Size([30])
    gt_bbox = targets["bboxes"] #torch.Size([30, 4])
    batch_idx = targets["batch_idx"] #torch.Size([30])
    # 还没加噪声的初始情况
    dn_cls = gt_cls.repeat(2 * dn_eve_box) #torch.Size([600])
    dn_bbox = gt_bbox.repeat(2 * dn_eve_box, 1) #torch.Size([600, 4])
    dn_batch_idx = batch_idx.repeat(2 * dn_eve_box).view(-1) #torch.Size([600])
    # 负样本索引
    neg_idx = torch.arange(total_num * dn_eve_box, dtype=torch.long, device=gt_bbox.device) + dn_eve_box * total_num #torch.Size([180])

    if cls_noise_ratio > 0:
        TF = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5) #随机百分之25类别替换
        noisy_idx = torch.nonzero(TF).squeeze(-1) #torch.Size([88])
        dn_cls[noisy_idx] = torch.randint(0, nc, (len(noisy_idx),), dtype=dn_cls.dtype, device=dn_cls.device)

    if box_noise_scale > 0:
        dn_bbox_xyxy = xywh2xyxy(dn_bbox) #torch.Size([600, 4])
        noise_scale = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale #根据描框大小生成噪声尺度权重

        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0 #torch.Size([360, 4]) 噪声方向
        rand_part = torch.rand_like(dn_bbox)
        rand_part[neg_idx] += 1.0 #负样本添加更多的噪声
        rand_sign = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0 #torch.Size([360, 4]) 噪声方向
        rand_part *= rand_sign #正负都可以

        dn_bbox_xyxy += rand_part * noise_scale #添加完噪声的bbox
        dn_bbox_xyxy.clip_(min=0.0, max=1.0) #限制边界框的取值范围
        dn_bbox = xyxy2xywh(dn_bbox_xyxy)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6) #增强器强度

    num_dn_double = int(batch_max_box * 2 * dn_eve_box) #对应一批里面的最大
    dn_cls_embed = class_embedding[dn_cls] #torch.Size([600, 256]) 加入噪声的类编码
    
    padding_cls = torch.zeros(batch_size, num_dn_double, dn_cls_embed.shape[-1], device=gt_cls.device) #torch.Size([3, 200, 256]) 批 总噪声 噪声类编码
    padding_bbox = torch.zeros(batch_size, num_dn_double, 4, device=gt_bbox.device)
    # torch.Size([3, 200, 4])
    num_dn_double_idx = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in gt_groups]) #torch.Size([30])
    pos_indices = torch.stack([num_dn_double_idx + batch_max_box * i for i in range(dn_eve_box)], dim=0) #torch.Size([10, 30])

    num_dn_double_idx = torch.cat([num_dn_double_idx + batch_max_box * i for i in range(2 * dn_eve_box)]) #torch.Size([600])
    padding_cls[(dn_batch_idx, num_dn_double_idx)] = dn_cls_embed
    padding_bbox[(dn_batch_idx, num_dn_double_idx)] = dn_bbox

    tgt_size = num_dn_double + num_queries
    attn_mask = torch.zeros([tgt_size, tgt_size], dtype=torch.bool)
    attn_mask[num_dn_double:, :num_dn_double] = True

    for i in range(dn_eve_box):
        if i == 0:
            attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), batch_max_box * 2 * (i + 1) : num_dn_double] = True
        elif i == dn_eve_box - 1:
            attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), : batch_max_box * 2 * i] = True
        else:
            attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), batch_max_box * 2 * (i + 1) : num_dn_double] = True
            attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), : batch_max_box * 2 * i] = True

    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in pos_indices.cpu().split(list(gt_groups), dim=1)],
        "dn_num_group": dn_eve_box,
        "dn_num_split": [num_dn_double, num_queries],
    }

    return (
        padding_cls.to(class_embedding.device),
        padding_bbox.to(class_embedding.device),
        attn_mask.to(class_embedding.device),
        dn_meta,
    )
    
