

#### IOU函数
```python
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False,FineSIoU= False, WIoU=False, Focal=False, pow=1, gamma=0.5, scale=False, eps=1e-7):
```


#### 标签分配
```python
class TaskAlignedAssigner(nn.Module):
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
```

#### bbox损失
```python
class BboxLoss(nn.Module):
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
```