# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh


class HungarianMatcher(nn.Module): #匈牙利匹配
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
        forward(pd_bboxes, pd_scores, gt_bboxes, gt_cls, bboxs_each_img, masks=None, gt_mask=None): Computes the
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

    def forward(self, pd_bboxes, pd_scores, gt_bboxes, gt_cls, bboxs_each_img, masks=None, gt_mask=None):
        """
        Forward pass for HungarianMatcher. This function computes costs based on prediction and ground truth
        (classification cost, L1 cost between boxes and GIoU cost between boxes) and finds the optimal matching between
        predictions and ground truth based on these costs.

        Args:
            pd_bboxes (Tensor): Predicted bounding boxes with shape [bs, nq, 4].
            pd_scores (Tensor): Predicted scores with shape [bs, nq, nc].
            gt_cls (torch.Tensor): Ground truth classes with shape [num_gts, ].
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape [num_gts, 4].
            bboxs_each_img (List[int]): List of length equal to batch size, containing the number of ground truths for
                each image.
            masks (Tensor, optional): Predicted masks with shape [bs, nq, height, width].
                Defaults to None.
            gt_mask (List[Tensor], optional): List of ground truth masks, each with shape [num_masks, Height, Width].
                Defaults to None.

        Returns:
            (List[Tuple[Tensor, Tensor]]): A list of size bs, each element is a tuple (index_i, index_j), where:
                - index_i is the tensor of indices of the selected predictions (in order)
                - index_j is the tensor of indices of the corresponding selected ground truth gt (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(nq, num_target_boxes)
        """

        bs, num_top_k, nc = pd_scores.shape

        if sum(bboxs_each_img) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # We flatten to compute the cost matrices in a batch
        # [bs * nq, nc]
        pd_scores = pd_scores.detach().view(-1, nc) #torch.Size([1200, 80])
        pd_scores = F.sigmoid(pd_scores) if self.use_fl else F.softmax(pd_scores, dim=-1) #torch.Size([1200, 80])
        # [bs * nq, 4]
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
        cost_bbox = (pd_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)  # (bs*nq, num_gt) #torch.Size([1200, 30, 4])->torch.Size([1200, 30])

        # Compute the GIoU cost between boxes, (bs*nq, num_gt)
        cost_giou = 1.0 - bbox_iou(pd_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1) #torch.Size([1200, 30])

        # Final cost matrix
        C = ( #torch.Size([1200, 30])
            self.cost_gain["class"] * cost_class
            + self.cost_gain["bbox"] * cost_bbox #l1损失
            + self.cost_gain["giou"] * cost_giou #giou损失
        ) 
        # Compute the mask cost and dice cost
        if self.with_mask:
            C += self._cost_mask(bs, bboxs_each_img, masks, gt_mask)

        # Set invalid values (NaNs and infinities) to 0 (fixes ValueError: matrix contains invalid numeric entries)
        C[C.isnan() | C.isinf()] = 0.0

        C = C.view(bs, num_top_k, -1).cpu() #torch.Size([4, 300, 30]) 批 topk 框
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(bboxs_each_img, -1))] #匈牙利匹配 返回预测框和真实框匹配索引
        bboxs_each_img = torch.as_tensor([0, *bboxs_each_img[:-1]]).cumsum_(0)  # (idx for queries, idx for gt)
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + bboxs_each_img[k])
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



        
# def generate_cdn_train_sample(gt, nc, nq, dn_cls_embed, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0, training=False):
#     if not training or num_dn <= 0:
#         return None, None, None, None
    
#     # bboxs_each_img
#     bboxs_each_img = gt["bboxs_each_img"] #[4, 7]
#     bs = len(bboxs_each_img) #2
#     batch_max_box = max(bboxs_each_img) #7
#     if batch_max_box == 0:
#         return None, None, None, None
#     batch_sum_bboxs = sum(bboxs_each_img) #11
#     num_dn_group = max(1, num_dn // batch_max_box) #长版效益


    
#     gt_cls = gt["cls"] #torch.Size([11])
#     gt_bbox = gt["bboxes"] #torch.Size([11, 4])
#     gt_img_idx = gt["img_idx"] #torch.Size([11])
#     # 还没加噪声的初始情况
#     dn_cls = gt_cls.repeat(2 * num_dn_group) #torch.Size([288])
#     dn_bbox = gt_bbox.repeat(2 * num_dn_group, 1) #torch.Size([288, 4])
#     dn_img_idx = gt_img_idx.repeat(2 * num_dn_group).view(-1) #torch.Size([288])
#     num_dn_double = int( 2 * num_dn_group * batch_max_box) #对应一批里面的最大
#     # 负样本索引
#     neg_sample_idx = torch.arange(batch_sum_bboxs * num_dn_group, dtype=torch.long, device=gt_bbox.device) + num_dn_group * batch_sum_bboxs #torch.Size([180])

#     if cls_noise_ratio > 0:
#         TF = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5) #随机百分之25类别替换
#         dn_cls_idx = torch.nonzero(TF).squeeze(-1) #torch.Size([88])
#         dn_cls[dn_cls_idx] = torch.randint(0, nc, (len(dn_cls_idx),), dtype=dn_cls.dtype, device=dn_cls.device)

#     if box_noise_scale > 0:
#         dn_bbox_xyxy = xywh2xyxy(dn_bbox) #torch.Size([600, 4])
#         scale_weight = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale #根据描框大小生成噪声尺度权重

#         rand_direction = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0 #torch.Size([360, 4]) 噪声方向
#         rand_weight = torch.rand_like(dn_bbox)
#         rand_weight[neg_sample_idx] += 1.0 #负样本添加更多的噪声
#         rand_direction = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0 #torch.Size([360, 4]) 噪声方向
#         rand_weight *= rand_direction #正负都可以

#         dn_bbox_xyxy += rand_weight * scale_weight #添加完噪声的bbox
#         dn_bbox_xyxy.clip_(min=0.0, max=1.0) #限制边界框的取值范围
#         dn_bbox = xyxy2xywh(dn_bbox_xyxy)
#         dn_bbox = torch.logit(dn_bbox, eps=1e-6) #增强其强度

    
#     dn_cls_embed = dn_cls_embed[dn_cls] #torch.Size([288, 256]) 加入噪声的类编码
    
#     dn_cls_embed = torch.zeros(bs, num_dn_double, dn_cls_embed.shape[-1], device=gt_cls.device) #torch.Size([2, 196, 256]) 批 总噪声 噪声类编码
#     dn_bboxs_embed = torch.zeros(bs, num_dn_double, 4, device=gt_bbox.device)
#     # torch.Size([2, 198, 4])
#     dn_pos = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in bboxs_each_img]) #torch.Size([12])
#     dn_pos_idx = torch.stack([dn_pos + batch_max_box * i for i in range(num_dn_group)], dim=0) #torch.Size([12, 11])

#     dn_pos = torch.cat([dn_pos + batch_max_box * i for i in range(2 * num_dn_group)]) #torch.Size([288])
#     dn_cls_embed[(dn_img_idx, dn_pos)] = dn_cls_embed #长版效益
#     dn_bboxs_embed[(dn_img_idx, dn_pos)] = dn_bbox

#     attn_mask_size = num_dn_double + nq
#     attn_mask = torch.zeros([attn_mask_size, attn_mask_size], dtype=torch.bool)
#     attn_mask[num_dn_double:, :num_dn_double] = True

#     for i in range(num_dn_group):
#         if i == 0:
#             attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), batch_max_box * 2 * (i + 1) : num_dn_double] = True
#         elif i == num_dn_group - 1:
#             attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), : batch_max_box * 2 * i] = True
#         else:
#             attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), batch_max_box * 2 * (i + 1) : num_dn_double] = True
#             attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), : batch_max_box * 2 * i] = True

#     dn_meta = {
#         "dn_pos_idx": [p.reshape(-1) for p in dn_pos_idx.cpu().split(list(bboxs_each_img), dim=1)], 
#         "num_dn_group": num_dn_group,
#         "dn_query_split": [num_dn_double, nq],
#     }

#     return (
#         dn_cls_embed.to(dn_cls_embed.device),
#         dn_bboxs_embed.to(dn_cls_embed.device),
#         attn_mask.to(dn_cls_embed.device),
#         dn_meta,
#     )
    




def generate_cdn_train_sample(gt, nc, nq, dn_embed, num_dn=100, cls_noise_ratio=0.5, box_noise_scale=1.0, training=False):
    if not training or num_dn <= 0:
        return None, None, None, None
    
    # bboxs_each_img
    bboxs_each_img = gt["bboxs_each_img"]  # [4, 8]
    bs = len(bboxs_each_img)  # 2
    batch_max_box = max(bboxs_each_img)  # 8
    if batch_max_box == 0:
        return None, None, None, None
    batch_sum_bboxs = sum(bboxs_each_img)  # 12
    num_dn_group = max(1, num_dn // batch_max_box)  # 12 长版效益

    gt_cls = gt["cls"]  # torch.Size([12])
    gt_bbox = gt["bboxes"]  # torch.Size([12, 4])
    gt_img_idx = gt["img_idx"]  # torch.Size([12])
    
    # 有效
    dn_cls = gt_cls.repeat(2 * num_dn_group)  # torch.Size([288])
    dn_bbox = gt_bbox.repeat(2 * num_dn_group, 1)  # torch.Size([288, 4])
    dn_img_idx = gt_img_idx.repeat(2 * num_dn_group).view(-1)  # torch.Size([288])
    
    
    
    num_dn_double = int(2 * num_dn_group * batch_max_box)  # 对应一批里面的最大

    # 负样本索引
    neg_sample_idx = torch.arange(batch_sum_bboxs * num_dn_group, dtype=torch.long, device=gt_bbox.device) + num_dn_group * batch_sum_bboxs  # torch.Size([144])

    # 添加类别噪声
    if cls_noise_ratio > 0:
        TF = torch.rand(dn_cls.shape) < (cls_noise_ratio * 0.5)  # 随机百分之25类别替换
        dn_cls_idx = torch.nonzero(TF).squeeze(-1)  # torch.Size([88])
        dn_cls[dn_cls_idx] = torch.randint(0, nc, (len(dn_cls_idx),), dtype=dn_cls.dtype, device=dn_cls.device)

    # 添加边界框噪声
    if box_noise_scale > 0:
        dn_bbox_xyxy = xywh2xyxy(dn_bbox)  # torch.Size([600, 4])
        scale_weight = (dn_bbox[..., 2:] * 0.5).repeat(1, 2) * box_noise_scale  # 根据描框大小生成噪声尺度权重
        rand_direction = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0  # torch.Size([360, 4]) 噪声方向
        rand_weight = torch.rand_like(dn_bbox)
        rand_weight[neg_sample_idx] += 1.0  # 负样本添加更多的噪声
        rand_direction = torch.randint_like(dn_bbox, 0, 2) * 2.0 - 1.0  # torch.Size([360, 4]) 噪声方向
        rand_weight *= rand_direction  # 正负都可以
        dn_bbox_xyxy += rand_weight * scale_weight  # 添加完噪声的bbox
        dn_bbox_xyxy.clip_(min=0.0, max=1.0)  # 限制边界框的取值范围
        dn_bbox = xyxy2xywh(dn_bbox_xyxy)
        dn_bbox = torch.logit(dn_bbox, eps=1e-6)  # 增强其强度

    # 噪声类编码与边界框编码
    dn_embed = dn_embed[dn_cls]  # torch.Size([288, 256]) 加入噪声的类编码
    dn_cls_embed = torch.zeros(bs, num_dn_double, dn_embed.shape[-1], device=gt_cls.device)  # torch.Size([2, 196, 256]) 批 总噪声 噪声类编码
    dn_bboxs_embed = torch.zeros(bs, num_dn_double, 4, device=gt_bbox.device)

    # 构造位置索引
    dn_pos = torch.cat([torch.tensor(range(num), dtype=torch.long) for num in bboxs_each_img])  # torch.Size([12])
    dn_pos_idx = torch.stack([dn_pos + batch_max_box * i for i in range(num_dn_group)], dim=0)  # torch.Size([12, 12])
    dn_pos = torch.cat([dn_pos + batch_max_box * i for i in range(2 * num_dn_group)])  # torch.Size([288])
    dn_cls_embed[(dn_img_idx, dn_pos)] = dn_embed  # 长版效益
    dn_bboxs_embed[(dn_img_idx, dn_pos)] = dn_bbox

    # 构建注意力掩码
    attn_mask_size = num_dn_double + nq
    attn_mask = torch.zeros([attn_mask_size, attn_mask_size], dtype=torch.bool)
    attn_mask[num_dn_double:, :num_dn_double] = True

    for i in range(num_dn_group):
        if i == 0:
            attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), batch_max_box * 2 * (i + 1) : num_dn_double] = True
        elif i == num_dn_group - 1:
            attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), : batch_max_box * 2 * i] = True
        else:
            attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), batch_max_box * 2 * (i + 1) : num_dn_double] = True
            attn_mask[batch_max_box * 2 * i : batch_max_box * 2 * (i + 1), : batch_max_box * 2 * i] = True

    # 构建元信息
    dn_meta = {
        "dn_pos_idx": [p.reshape(-1) for p in dn_pos_idx.cpu().split(list(bboxs_each_img), dim=1)],  # 最多的bbox图像满了
        "num_dn_group": num_dn_group,
        "dn_query_split": [num_dn_double, nq],
    }

    return (
        dn_cls_embed.to(dn_cls_embed.device),
        dn_bboxs_embed.to(dn_cls_embed.device),
        attn_mask.to(dn_cls_embed.device),
        dn_meta,
    )
