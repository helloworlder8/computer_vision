import numpy as np
import torch
import math
from ultralytics.utils import ops
import pickle

class WIoU_Scale:
    ''' monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
        momentum: The momentum of running mean'''
    iou_mean = 1.
    monotonous = False
    _momentum = 1 - 0.5 ** (1 / 7000)
    _is_train = True

    def __init__(self, iou):
        self.iou = iou
        self._update(self)

    @classmethod
    def _update(cls, self):
        if cls._is_train: cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, WIoU=False, ShapeIoU=False,
             hw=1, mpdiou=False, Inner=False, alpha=1, ratio=0.7, eps=1e-7, scale=0.0):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        EIoU (bool, optional): If True, calculate Efficient IoU. Defaults to False.
        SIoU (bool, optional): If True, calculate Scylla IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """
    if Inner:
        if not xywh:
            box1, box2 = ops.xyxy2xywh(box1), ops.xyxy2xywh(box2)
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - (w1 * ratio) / 2, x1 + (w1 * ratio) / 2, y1 - (h1 * ratio) / 2, y1 + (
                    h1 * ratio) / 2
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - (w2 * ratio) / 2, x2 + (w2 * ratio) / 2, y2 - (h2 * ratio) / 2, y2 + (
                    h2 * ratio) / 2

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

        # Union Area
        union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
        iou = inter / union
    # Get the coordinates of bounding boxes
    else:
        if xywh:  # transform from xywh to xyxy
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1) #torch.Size([91, 1])
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1) #torch.Size([91, 1])
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps #每一个bbox的宽和高
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area  交集区域
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
        # Union Area 并集区域
        union = w1 * h1 + w2 * h2 - inter + eps
        # IoU
        iou = inter / union #交并比

    if CIoU or DIoU or GIoU or EIoU or SIoU or ShapeIoU or mpdiou or WIoU:
        convex_width = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        convex_height = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or mpdiou or WIoU or ShapeIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            convex_diagonal_squared = convex_width ** 2 + convex_height ** 2 + eps  # convex diagonal squared
            center_dist_squared = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (center_dist_squared / convex_diagonal_squared + v * alpha)  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = convex_width ** 2 + eps
                ch2 = convex_height ** 2 + eps
                return iou - (center_dist_squared / convex_diagonal_squared + rho_w2 / cw2 + rho_h2 / ch2) # EIoU
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / convex_width) ** 2
                rho_y = (s_ch / convex_height) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - 0.5 * (distance_cost + shape_cost) + eps # SIoU
            elif ShapeIoU:
                #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance
                ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                convex_width = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
                convex_height = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
                convex_diagonal_squared = convex_width ** 2 + convex_height ** 2 + eps                            # convex diagonal squared
                center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
                center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                center_distance = hh * center_distance_x + ww * center_distance_y
                distance = center_distance / convex_diagonal_squared

                #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape
                omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - distance - 0.5 * shape_cost
            elif mpdiou:
                d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
                d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
                return iou - d1 / hw.unsqueeze(1) - d2 / hw.unsqueeze(1)  # MPDIoU
            elif WIoU:
                self = WIoU_Scale(1 - iou)
                dist = getattr(WIoU_Scale, '_scaled_loss')(self)
                return iou * dist  # WIoU https://arxiv.org/abs/2301.10051
            return iou - center_dist_squared / convex_diagonal_squared  # DIoU
        c_area = convex_width * convex_height + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def test_bbox_iou():
    # 创建两个包围框张量
    with open('runs/debug_param/box1.pkl', 'rb') as f:
        box1 = pickle.load(f)
    with open('runs/debug_param/box2.pkl', 'rb') as f:
        box2 = pickle.load(f)

    # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True) #x1 y1 x2 y2用的两顶点坐标
    # 计算 IoU
    iou = bbox_iou(box1, box2, xywh=False, ShapeIoU=True)

    print("IoU values:")
    print(iou.shape)

if __name__ == "__main__":
    test_bbox_iou()