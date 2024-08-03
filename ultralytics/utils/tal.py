# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn

from  ultralytics.utils.checks import check_version
from  ultralytics.utils.metrics import bbox_iou, probiou
from  ultralytics.utils.ops import xywhr2xyxyxyxy

# from .checks import check_version
# from .metrics import bbox_iou, probiou
# from .ops import xywhr2xyxyxyxy

TORCH_1_10 = check_version(torch.__version__, "1.10.0")


class TaskAlignedAssigner(nn.Module):

    def __init__(self, topk=13, num_class=80, alpha=1.0, beta=6.0, eps=1e-9):
        """Initialize a TaskAlignedAssigner object with customizable hyperparameters."""
        super().__init__()
        self.topk = topk ## 每个gt box最多选择topk个候选框作为正样本 10
        self.num_class = num_class #80
        self.bg_idx = num_class #80
        self.alpha = alpha #0.5
        self.beta = beta #6
        self.eps = eps #极小


        # _, target_bboxes_BA4matrix, mask_target_class_BACmatrix, mask_poss_sum_bool, _ = self.assigner.forward(
        #     anchor_points * stride_points, #      8400 2

        #     pd_class.detach().sigmoid(), #torch.Size([2, 8400, 80])   sigmoid
        #     (pd_bboxes.detach() * stride_points).type(gt_bboxes.dtype), #torch.Size([2, 8400, 4])

        #     gt_class, #torch.Size([2, 8, 1])
        #     gt_bboxes, #torch.Size([2, 8, 4])
        #     mask_gt_BT1,   #torch.Size([2, 8, 1])
        # )
    @torch.no_grad() 
    def forward(self, anchor_points, pd_class, pd_bboxes,  gt_class, gt_bboxes, mask_gt_BT1):
        # tensors = {
        #     'anchor_points': anchor_points,
        
        #     'pd_class': pd_class,
        #     'pd_bboxes': pd_bboxes,

        #     'gt_class': gt_class,
        #     'gt_bboxes': gt_bboxes,
        #     'mask_gt_BT1': mask_gt_BT1,
        # }
        # torch.save(tensors, 'runs/debug_param/TaskAlignedAssigne.pt')

        # 标签分配
        self.batch_size = gt_bboxes.shape[0] #2
        self.num_max_boxes = gt_bboxes.shape[1] #8

        # 如果不存在真实框，直接返回结果
        if self.num_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_class[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_class).to(device),
                torch.zeros_like(pd_class[..., 0]).to(device),
                torch.zeros_like(pd_class[..., 0]).to(device),
            )

        mask_pos_BTAmetrics, fuse_BTAmaxtric, gt_bboxes_score_BTAmaxtric = self.computer_maskpos_fusemetric_bboxesscores(
            anchor_points, pd_class, pd_bboxes, gt_class, gt_bboxes, mask_gt_BT1
        )
        # 压缩给索引     压缩填1
        mask_poss_idx_BAmetrics, mask_poss_sum_BAmetrics, mask_pos_BTAmetrics = self.mask_pos_postprocessing(mask_pos_BTAmetrics, gt_bboxes_score_BTAmaxtric, self.num_max_boxes)

        # Assigned target
        target_class_BAmatrix, target_bboxes_BA4matrix, mask_target_class_BACmatrix = self.get_targets(gt_class, gt_bboxes, mask_poss_idx_BAmetrics, mask_poss_sum_BAmetrics)

        # Normalize
        fuse_BTAmaxtric *= mask_pos_BTAmetrics #torch.Size([2, 8, 8400])
        pos_align_metrics = fuse_BTAmaxtric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_bboxes_scores = (gt_bboxes_score_BTAmaxtric * mask_pos_BTAmetrics).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (fuse_BTAmaxtric * pos_bboxes_scores / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        mask_target_class_BACmatrix = mask_target_class_BACmatrix * norm_align_metric

        return target_class_BAmatrix, target_bboxes_BA4matrix, mask_target_class_BACmatrix, mask_poss_sum_BAmetrics.bool(), mask_poss_idx_BAmetrics




                 #                                 ([8400, 2])([2, 8400, 80])([2, 8400, 4])([2, 8, 1])([2, 8, 4])([2, 8, 1])
    def computer_maskpos_fusemetric_bboxesscores(self,anchor_points, pd_class, pd_bboxes, gt_class, gt_bboxes, mask_gt_BT1):
        """Get in_gts mask, (b, max_num_obj, h*w).""" 

        """ 边界条件过滤 """
        mask_gt_BTA = self.boundary_condition_filtering(anchor_points, gt_bboxes) #边界过滤 2 8400 4
        """ 计算融合得分矩阵和边界得分矩阵"""                                                   
        fuse_BTAmaxtric, gt_bboxes_score_BTAmaxtric = self.computer_fuse_BTAmaxtric(pd_class, pd_bboxes, gt_class, gt_bboxes, mask_gt_BTA * mask_gt_BT1) #
        """一个target 最多10个点 """
        mask_tk_fuse_BTAmetrics = self.get_best_topk_form_fuse_BTAmetrics_mask(fuse_BTAmaxtric, topk_mask=mask_gt_BT1.expand(-1, -1, self.topk).bool()) #torch.Size([2, 8, 8400])  0 1
        """ 综合条件 有目标 边界满足 值在前10 """
        mask_pos_BTAmetrics = mask_gt_BT1 * mask_gt_BTA * mask_tk_fuse_BTAmetrics  #前10综合得分前10  anchir_point在内并且靠中心  最基本要求  递进关系
        # torch.equal(mask_tk_fuse_BTAmetrics, mask_pos_BTAmetrics)

        return mask_pos_BTAmetrics, fuse_BTAmaxtric, gt_bboxes_score_BTAmaxtric #torch.Size([2, 8, 8400]) torch.Size([2, 8, 8400]) torch.Size([2, 8, 8400])



    #                                             80        4          8 1      8  4           2 8 8400
    def computer_fuse_BTAmaxtric(self, pd_class, pd_bboxes, gt_class, gt_bboxes, mask_gt_BTA): #torch.Size([2, 8, 8400])
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2] #8400 
        mask_gt_BTA = mask_gt_BTA.bool()  # 2 8 8400
        gt_class_BTAmaxtric = torch.zeros([self.batch_size, self.num_max_boxes, na], dtype=pd_class.dtype, device=pd_class.device)#2 8 8400
        gt_bboxes_score_BTAmaxtric = torch.zeros([self.batch_size, self.num_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device) #2 8 8400

        # 获得预测类信息
        ind = torch.zeros([2, self.batch_size, self.num_max_boxes], dtype=torch.long)  # 2, 2 8
        ind[0] = torch.arange(end=self.batch_size).view(-1, 1).expand(-1, self.num_max_boxes)  # 2 8 第几幅画
        ind[1] = gt_class.squeeze(-1)                                                          # 2 8 第几类
        gt_class_BTAmaxtric[mask_gt_BTA] = pd_class[ind[0], :, ind[1]][mask_gt_BTA]  # b, max_num_obj, h*w torch.Size([2, 8, 8400]) 没有赋值
        
        
        # 预测和真实的iou
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.num_max_boxes, -1, -1)[mask_gt_BTA] #([2, 1, 8400, 4])->([3392, 4])
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt_BTA] #->torch.Size([2, 8, 8400, 4])->([3392, 4])
        gt_bboxes_score_BTAmaxtric[mask_gt_BTA] = self.iou_calculation(gt_boxes, pd_boxes)


        # 相当于在回归问题上用的了预测类的信息
        fuse_BTAmaxtric = gt_class_BTAmaxtric.pow(self.alpha) * gt_bboxes_score_BTAmaxtric.pow(self.beta)
        return fuse_BTAmaxtric, gt_bboxes_score_BTAmaxtric

    def iou_calculation(self, gt_bboxes, pd_bboxes): #主逻辑改动
        """IoU calculation for horizontal bounding boxes."""
        # return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, SIoU=True).squeeze(-1).clamp_(0)
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def get_best_topk_form_fuse_BTAmetrics_mask(self, fuse_BTAmetrics, largest=True, topk_mask=None): #2 8 10

        # (b, max_num_obj, topk 看看哪个点预测得比较好
        topk_metrics, topk_idxs = torch.topk(fuse_BTAmetrics, self.topk, dim=-1, largest=largest) #([2, 8, 8400])->([2, 8, 10])
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)

        topk_idxs.masked_fill_(~topk_mask, 0) #torch.Size([2, 8, 10])

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        mask_tk_fuse_BTAmetrics = torch.zeros(fuse_BTAmetrics.shape, dtype=torch.int8, device=topk_idxs.device) #torch.Size([2, 8, 8400])
        ones_like = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device) #torch.Size([2, 8, 1])
        for k in range(self.topk):
            # 在哪个点上打1
            mask_tk_fuse_BTAmetrics.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones_like)
        # 有2标记为0
        mask_tk_fuse_BTAmetrics.masked_fill_(mask_tk_fuse_BTAmetrics > 1, 0) #过滤重复的索引

        return mask_tk_fuse_BTAmetrics.to(fuse_BTAmetrics.dtype)

    def get_targets(self, gt_class, gt_bboxes, mask_poss_idx_BAmetrics, mask_poss_sum_BAmetrics):



        batch_ind = torch.arange(end=self.batch_size, dtype=torch.int64, device=gt_class.device)[..., None]
        mask_poss_target_idx_BAmetrics = mask_poss_idx_BAmetrics + batch_ind * self.num_max_boxes  # ([2, 8400]) ([2, 1]) 8
        target_class_BAmatrix = gt_class.long().flatten()[mask_poss_target_idx_BAmetrics]  #16 ([2, 8400])->([2, 8400])
        # torch.Size([2, 8400])  16
        # 合并选择
        target_bboxes_BA4matrix = gt_bboxes.view(-1, gt_bboxes.shape[-1])[mask_poss_target_idx_BAmetrics] #集合bbox操作 ([2, 8400, 4])
        # torch.Size([2, 8400, 4])  16 4
        # Assigned target scores
        target_class_BAmatrix.clamp_(0)

        # 10x faster than F.one_hot()
        mask_target_class_BACmatrix = torch.zeros(
            (target_class_BAmatrix.shape[0], target_class_BAmatrix.shape[1], self.num_class),
            dtype=torch.int64,
            device=target_class_BAmatrix.device,
        )  # (b, h*w, 80)
        mask_target_class_BACmatrix.scatter_(2, target_class_BAmatrix.unsqueeze(-1), 1) # torch.Size([2, 8400]) torch.Size([2, 8400, 80])

        mask_class = mask_poss_sum_BAmetrics[:, :, None].repeat(1, 1, self.num_class)  # (b, h*w, 80)
        mask_target_class_BACmatrix = torch.where(mask_class > 0, mask_target_class_BACmatrix, 0)

        return target_class_BAmatrix, target_bboxes_BA4matrix, mask_target_class_BACmatrix # 2 8400   2 8400 4   2 8400 80

    @staticmethod
    def boundary_condition_filtering(anchor_points, gt_bboxes, eps=1e-9): #([8400, 2]) shape(2, 8, 4)

        num_anchors = anchor_points.shape[0] #8400
        batch_size, num_bboxes, _ = gt_bboxes.shape #2 8 4
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # ([16, 1, 2])     0 8400 2
        mask_gt_BA4bboxes = torch.cat((anchor_points[None] - lt, rb - anchor_points[None]), dim=2)
        mask_gt_BA4bboxes = mask_gt_BA4bboxes.view(batch_size, num_bboxes, num_anchors, -1) #2 8 8400 4
        # torch.Size([2, 8, 8400, 4])
        return mask_gt_BA4bboxes.amin(3).gt_(eps) #torch.Size([2, 8, 8400]) 图目标点

    @staticmethod
    def mask_pos_postprocessing(mask_pos_BTAmetrics, gt_bboxes_score_BTAmaxtric, num_max_boxes):
        # (b, num_max_boxes, h*w) -> (b, h*w)
        mask_poss_sum_BAmetrics = mask_pos_BTAmetrics.sum(-2) #torch.Size([2, 8400])  不管哪个目标只要有点就行
        if mask_poss_sum_BAmetrics.max() > 1:  # 一点多框
            mask_multi_gts = (mask_poss_sum_BAmetrics.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)  # (b, num_max_boxes, h*w)
            max_bboxes_scores_idx = gt_bboxes_score_BTAmaxtric.argmax(1)  # (b, h*w)

            is_max_bboxes_scores = torch.zeros(mask_pos_BTAmetrics.shape, dtype=mask_pos_BTAmetrics.dtype, device=mask_pos_BTAmetrics.device)
            is_max_bboxes_scores.scatter_(1, max_bboxes_scores_idx.unsqueeze(1), 1)

            mask_pos_BTAmetrics = torch.where(mask_multi_gts, is_max_bboxes_scores, mask_pos_BTAmetrics).float()  # (b, num_max_boxes, h*w)
            mask_poss_sum_BAmetrics = mask_pos_BTAmetrics.sum(-2)
        # Find each grid serve which gt(index)
        mask_poss_idx_BAmetrics = mask_pos_BTAmetrics.argmax(-2)  # (b, h*w)  mask_pos预测和真实的综合结果
        return mask_poss_idx_BAmetrics, mask_poss_sum_BAmetrics, mask_pos_BTAmetrics


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """IoU calculation for rotated bounding boxes."""
        return probiou(gt_bboxes, pd_bboxes).squeeze(-1).clamp_(0)

    @staticmethod
    def boundary_condition_filtering(anchor_points, gt_bboxes):
        """
        Select the positive anchor center in gt for rotated bounding boxes.

        Args:
            anchor_points (Tensor): shape(h*w, 2)
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
        ap = anchor_points - a
        norm_ab = (ab * ab).sum(dim=-1)
        norm_ad = (ad * ad).sum(dim=-1)
        ap_dot_ab = (ap * ab).sum(dim=-1)
        ap_dot_ad = (ap * ad).sum(dim=-1)
        return (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)  # is_in_box


def make_anchors(preds, strides_list, offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_points = [], []
    assert preds is not None
    dtype, device = preds[0].dtype, preds[0].device #想要同等数据类型和设备
    for i, stride in enumerate(strides_list):
        _, _, h, w = preds[i].shape #宽高
        gridi = torch.arange(end=h, device=device, dtype=dtype) + offset  
        gridj = torch.arange(end=w, device=device, dtype=dtype) + offset  
        gridi, gridj = torch.meshgrid(gridi, gridj, indexing="ij") if TORCH_1_10 else torch.meshgrid(gridi, gridj)
        anchor_points.append(torch.stack((gridj, gridi), -1).view(-1, 2)) #torch.Size([80, 80, 2])竖y横x torch.Size([6400, 2])
        stride_points.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_points) #torch.Size([8400, 2])  torch.Size([8400, 1])和特征1图尺寸强相关


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):#torch.Size([2, 8400, 4]) torch.Size([8400, 2])
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim) #left top     right bottle
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max): #torch.Size([8400, 2]) torch.Size([2, 8400, 4])
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)  # dist (lt, rb)
# torch.Size([2, 8400, 4]) torch.Size([2, 8400, 4]) 范围限制

def dist2rbox(pred_dist, pred_angle, anchor_points, dim=-1):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.

    Args:
        pred_dist (torch.Tensor): Predicted rotated distance, (batch_size, h*w, 4).
        pred_angle (torch.Tensor): Predicted angle, (batch_size, h*w, 1).
        anchor_points (torch.Tensor): Anchor points, (h*w, 2).
    Returns:
        (torch.Tensor): Predicted rotated bounding boxes, (batch_size, h*w, 4).
    """
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    # (batch_size, h*w, 1)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x, y = xf * cos - yf * sin, xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)



if __name__ == '__main__':
     
    assigner =  TaskAlignedAssigner(10,80,0.5,6)
    tensors = torch.load('runs/debug_param/TaskAlignedAssigne.pt')
# def forward(self, pd_class, pd_bboxes, anchor_points, gt_class, gt_bboxes, mask_gt_BT1)
# 假设`assigner`是`TaskAlignedAssigner`的一个实例
    _, target_bboxes_BA4matrix, mask_target_class_BACmatrix, mask_poss_sum_BAmetrics, _ = assigner(tensors['anchor_points'],
                                                        tensors['pd_class'],
                                                        tensors['pd_bboxes'],                                                        
                                                        tensors['gt_class'],
                                                        tensors['gt_bboxes'],
                                                        tensors['mask_gt_BT1'])

