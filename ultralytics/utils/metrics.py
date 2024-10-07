# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings

OKS_SIGMA = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
    / 10.0
)


def bbox_ioa(box1, box2, iou=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes are in x1y1x2y2 format.

    Args:
        box1 (np.ndarray): A numpy array of shape (n, 4) representing n bounding boxes.
        box2 (np.ndarray): A numpy array of shape (m, 4) representing m bounding boxes.
        iou (bool): Calculate the standard IoU if True else return inter_area/box2_area.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (np.ndarray): A numpy array of shape (n, m) representing the intersection over box2 area.
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
        np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
    ).clip(0)

    # Box2 area
    area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    if iou:
        box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area = area + box1_area[:, None] - inter_area

    # Intersection over box2 area
    return inter_area / (area + eps)


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """

    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


# def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
#     """
#     Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

#     Args:
#         box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
#         box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
#         xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
#                                (x1, y1, x2, y2) format. Defaults to True.
#         GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
#         DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
#         CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
#         eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

#     Returns:
#         (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
#     """

#     # Get the coordinates of bounding boxes
#     if xywh:  # transform from xywh to xyxy
#         (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
#         w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
#         b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
#         b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
#     else:  # x1, y1, x2, y2 = box1
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
#         w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
#         w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

#     # Intersection area
#     inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
#         b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
#     ).clamp_(0)

#     # Union Area
#     union = w1 * h1 + w2 * h2 - inter + eps

#     # IoU
#     iou = inter / union
#     if CIoU or DIoU or GIoU:
#         cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
#         ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
#         if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
#             c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
#             rho2 = (
#                 (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
#             ) / 4  # center dist**2
#             if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
#                 v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
#                 with torch.no_grad():
#                     alpha = v / (v - iou + (1 + eps))
#                 return iou - (rho2 / c2 + v * alpha)  # CIoU
#             return iou - rho2 / c2  # DIoU
#         c_area = cw * ch + eps  # convex area
#         return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
#     return iou  # IoU



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
                                         cls._momentum * self.iou.detaconvex_hei().mean().item()
    
    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detaconvex_hei() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detaconvex_hei() / self.iou_mean
                pow = delta * torch.pow(gamma, beta - delta)
                return beta / pow
        return 1

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False,FineSIoU= False, WIoU=False, Focal=False, pow=1, gamma=0.5, scale=False, eps=1e-7):


    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1) #list(tensor(list))
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1) #torch.Size([53121, 4])->torch.Size([53121, 1])
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1) #torch.Size([53121, 4])->torch.Size([53121, 1])
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps #torch.Size([53121, 1])
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps #torch.Size([53121, 1])



    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)


    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    if scale:
        self = WIoU_Scale(1 - (inter / union))

    # IoU
    # iou = inter / union # ori iou
    iou = torch.pow(inter/(union + eps), pow) # pow iou

    if CIoU or DIoU or GIoU or EIoU or SIoU or FineSIoU or WIoU:
        convex_width = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        convex_height = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or FineSIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            convex_diagonal_squared = convex_width ** 2 + convex_height ** 2 + eps  # convex diagonal squared
            center_width = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps #宽中心点距离
            center_height = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps #高中心点距离
            center_dist_squared = torch.pow(center_width ** 2 + center_height ** 2, pow)  # 中心距离平方
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorconvex_hei/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                if Focal:
                    return iou - (center_dist_squared / convex_diagonal_squared + torch.pow(v * alpha + eps, pow)), torch.pow(inter/(union + eps), gamma)  # Focal_CIoU
                else:


                    """ CIoU """
                    return iou - (center_dist_squared / convex_diagonal_squared + torch.pow(v * alpha + eps, pow))  # CIoU
            elif EIoU:
                wid_dif2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2 #torch.Size([53121, 1])宽度差的平方
                hei_dif2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2 #torch.Size([53121, 1])高度差的平方
                convex_wid2 = torch.pow(convex_width ** 2 + eps, pow)
                convex_hei2 = torch.pow(convex_height ** 2 + eps, pow)
                if Focal:
                    return iou - (center_dist_squared / convex_diagonal_squared + wid_dif2 / convex_wid2 + hei_dif2 / convex_hei2), torch.pow(inter/(union + eps), gamma) # Focal_EIou
                else:
                    return iou - (center_dist_squared / convex_diagonal_squared + wid_dif2 / convex_wid2 + hei_dif2 / convex_hei2) # EIou


            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                center_dist = torch.pow(center_width ** 2 + center_height ** 2, 0.5) #计算中心点距离
                sin_pow_1 = torch.abs(center_width) / center_dist #宽比距离
                sin_pow_2 = torch.abs(center_height) / center_dist #高比距离
                threshold = 2 ** 0.5 / 2
                sin_pow = torch.where(sin_pow_1 > threshold, sin_pow_2, sin_pow_1) #宽高正弦值选择

                angle_cost = torch.cos(torch.arcsin(sin_pow) * 2 - math.pi / 2) #角度成本


                rho_x = (center_width / convex_width) ** 2 #距离成本
                rho_y = (center_height / convex_height) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)


                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2) #形状成本
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)

                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                if Focal:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, pow), torch.pow(inter/(union + eps), gamma) # Focal_SIou
                else:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, pow) # SIou
                
                
            elif FineSIoU:


                # shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                # return iou - 0.5 * (distance_cost + shape_cost) + eps # SIoU
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                center_dist = torch.pow(center_width ** 2 + center_height ** 2, 0.5) #中心点距离


                """ #角度成本 """
                sin_w = torch.abs(center_width) / center_dist #中心点距离宽比中心点距离
                sin_h = torch.abs(center_height) / center_dist #中心点距离高比中心点距离
                threshold = 2 ** 0.5 / 2    #阈值
                sin_best = torch.where(sin_w > threshold, sin_h, sin_w) #根据角度选合适的

                angle_param = torch.cos(torch.arcsin(sin_best) * 2 - math.pi / 2)        #角度成本 0 -》1 1》0
                angle_cost = torch.pow(1 - torch.exp(-(0.9847-angle_param)),3)  #5度 角度成本系数

                """ 距离成本  原始  中心差比包围"""

                rho_x1 = (center_width / convex_width) ** 2
                rho_y1 = (center_height / convex_height) ** 2
                # rho_x = (center_width / min_w) ** 2
                # rho_y = (center_height / min_h) ** 2
                gamma =  2 -angle_param
                distance_cost = 2 - torch.exp(-gamma * rho_x1) - torch.exp(-gamma * rho_y1)


                """ 形状成本    差值比最大"""
                omiga_w = torch.abs(w1 - w2) / w1 #两者宽差比宽最大的
                omiga_h = torch.abs(h1 - h2) / h1
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 6) + torch.pow(1 - torch.exp(-1 * omiga_h), 6) #形状成本系数
                if Focal:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, pow), torch.pow(inter/(union + eps), gamma) # Focal_SIou
                else:
                    return iou - torch.pow(0.5 * (angle_cost + distance_cost + shape_cost) + eps, pow) # SIou
            
            
            elif WIoU:
                if Focal:
                    raise RuntimeError("WIoU do not support Focal.")
                elif scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(self), (1 - iou) * torch.exp((center_dist_squared / convex_diagonal_squared)), iou # WIoU https://arxiv.org/abs/2301.10051
                else:
                    return iou, torch.exp((center_dist_squared / convex_diagonal_squared)) # WIoU v1
            if Focal:
                return iou - center_dist_squared / convex_diagonal_squared, torch.pow(inter/(union + eps), gamma)  # Focal_DIoU
            else:


                """ DIoU """
                return iou - center_dist_squared / convex_diagonal_squared  # DIoU
        c_area = center_width * center_height + eps  # convex area
        if Focal:
            return iou - torch.pow((c_area - union) / c_area + eps, pow), torch.pow(inter/(union + eps), gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
        else:
            return iou - torch.pow((c_area - union) / c_area + eps, pow)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    if Focal:
        return iou, torch.pow(inter/(union + eps), gamma)  # Focal_IoU
    else:
        return iou  # IoU



def get_inner_iou(box1, box2, xywh=True, eps=1e-7, ratio=0.7):
    if not xywh:
        box1, box2 = ops.xyxy2xywh(box1), ops.xyxy2xywh(box2)
    (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - (w1 * ratio) / 2, x1 + (w1 * ratio) / 2, y1 - (h1 * ratio) / 2, y1 + (h1 * ratio) / 2
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - (w2 * ratio) / 2, x2 + (w2 * ratio) / 2, y2 - (h2 * ratio) / 2, y2 + (h2 * ratio) / 2
    
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 * ratio * ratio + w2 * h2 * ratio * ratio - inter + eps
    return inter / union

def bbox_inner_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, ShapeIoU=False, eps=1e-7, ratio=0.7, scale=0.0):
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

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    innner_iou = get_inner_iou(box1, box2, xywh=xywh, ratio=ratio)
    
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or EIoU or SIoU or ShapeIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return innner_iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return innner_iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2) # EIoU
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
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return innner_iou - 0.5 * (distance_cost + shape_cost) + eps # SIoU
            elif ShapeIoU:
                #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance  
                ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
                ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
                c2 = cw ** 2 + ch ** 2 + eps                            # convex diagonal squared
                center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
                center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
                center_distance = hh * center_distance_x + ww * center_distance_y
                distance = center_distance / c2

                #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    
                omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return innner_iou - distance - 0.5 * shape_cost
            return innner_iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return innner_iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return innner_iou  # IoU



def bbox_mpdiou(box1, box2, xywh=True, mpdiou_hw=1, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # IoU
    iou = inter / union
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return iou - d1 / mpdiou_hw.unsqueeze(1) - d2 / mpdiou_hw.unsqueeze(1)  # MPDIoU

def bbox_inner_mpdiou(box1, box2, xywh=True, mpdiou_hw=1, ratio=0.7, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Inner-IoU
    innner_iou = get_inner_iou(box1, box2, xywh=xywh, ratio=ratio)
    
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
    d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
    return innner_iou - d1 / mpdiou_hw.unsqueeze(1) - d2 / mpdiou_hw.unsqueeze(1)  # MPDIoU

def wasserstein_loss(pred, target, eps=1e-7, constant=12.8):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x_min, y_min, x_max, y_max),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    b1_x1, b1_y1, b1_x2, b1_y2 = pred.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    
    b1_x_center, b1_y_center = b1_x1 + w1 / 2, b1_y1 + h1 / 2
    b2_x_center, b2_y_center = b2_x1 + w2 / 2, b2_y1 + h2 / 2
    center_distance = (b1_x_center - b2_x_center) ** 2 + (b1_y_center - b2_y_center) ** 2 + eps
    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)



class WiseIouLoss(torch.nn.Module):
    ''' :param monotonous: {
            None: origin V1
            True: monotonic FM V2
            False: non-monotonic FM V3
        }'''
    momentum = 1e-2
    alpha = 1.7
    delta = 2.7

    def __init__(self, ltype='WIoU', monotonous=False, inner_iou=False):
        super().__init__()
        assert getattr(self, f'_{ltype}', None), f'The loss function {ltype} does not exist'
        self.ltype = ltype
        self.monotonous = monotonous
        self.inner_iou = inner_iou
        self.register_buffer('iou_mean', torch.tensor(1.))

    def __getitem__(self, item):
        if callable(self._fget[item]):
            self._fget[item] = self._fget[item]()
        return self._fget[item]

    def forward(self, pred, target, ret_iou=False, ratio=1.0, **kwargs):
        self._fget = {
            # pred, target: x0,y0,x1,y1
            'pred': pred,
            'target': target,
            # x,y,w,h
            'pred_xy': lambda: (self['pred'][..., :2] + self['pred'][..., 2: 4]) / 2,
            'pred_wh': lambda: self['pred'][..., 2: 4] - self['pred'][..., :2],
            'target_xy': lambda: (self['target'][..., :2] + self['target'][..., 2: 4]) / 2,
            'target_wh': lambda: self['target'][..., 2: 4] - self['target'][..., :2],
            # x0,y0,x1,y1
            'min_coord': lambda: torch.minimum(self['pred'][..., :4], self['target'][..., :4]),
            'max_coord': lambda: torch.maximum(self['pred'][..., :4], self['target'][..., :4]),
            # The overlapping region
            'wh_inter': lambda: torch.relu(self['min_coord'][..., 2: 4] - self['max_coord'][..., :2]),
            's_inter': lambda: torch.prod(self['wh_inter'], dim=-1),
            # The area covered
            's_union': lambda: torch.prod(self['pred_wh'], dim=-1) +
                               torch.prod(self['target_wh'], dim=-1) - self['s_inter'],
            # The smallest enclosing box
            'wh_box': lambda: self['max_coord'][..., 2: 4] - self['min_coord'][..., :2],
            's_box': lambda: torch.prod(self['wh_box'], dim=-1),
            'l2_box': lambda: torch.square(self['wh_box']).sum(dim=-1),
            # The central points' connection of the bounding boxes
            'd_center': lambda: self['pred_xy'] - self['target_xy'],
            'l2_center': lambda: torch.square(self['d_center']).sum(dim=-1),
            # IoU / Inner-IoU
            'iou': (1 - get_inner_iou(pred, target, xywh=False, ratio=ratio).squeeze()) if self.inner_iou else lambda: 1 - self['s_inter'] / self['s_union'],
        }

        if self.training:
            self.iou_mean.mul_(1 - self.momentum)
            self.iou_mean.add_(self.momentum * self['iou'].detach().mean())

        ret = self._scaled_loss(getattr(self, f'_{self.ltype}')(**kwargs)), self['iou']
        delattr(self, '_fget')
        return ret if ret_iou else ret[0]

    def _scaled_loss(self, loss, iou=None):
        if isinstance(self.monotonous, bool):
            beta = (self['iou'].detach() if iou is None else iou) / self.iou_mean

            if self.monotonous:
                loss *= beta.sqrt()
            else:
                divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
                loss *= beta / divisor
        return loss

    def _IoU(self):
        return self['iou']

    def _WIoU(self):
        dist = torch.exp(self['l2_center'] / self['l2_box'].detach())
        return dist * self['iou']

    def _EIoU(self):
        penalty = self['l2_center'] / self['l2_box'] \
                  + torch.square(self['d_center'] / self['wh_box']).sum(dim=-1)
        return self['iou'] + penalty

    def _GIoU(self):
        return self['iou'] + (self['s_box'] - self['s_union']) / self['s_box']

    def _DIoU(self):
        return self['iou'] + self['l2_center'] / self['l2_box']

    def _CIoU(self, eps=1e-4):
        v = 4 / math.pi ** 2 * \
            (torch.atan(self['pred_wh'][..., 0] / (self['pred_wh'][..., 1] + eps)) -
             torch.atan(self['target_wh'][..., 0] / (self['target_wh'][..., 1] + eps))) ** 2
        alpha = v / (self['iou'] + v)
        return self['iou'] + self['l2_center'] / self['l2_box'] + alpha.detach() * v

    def _SIoU(self, theta=4):
        # Angle Cost
        angle = torch.arcsin(torch.abs(self['d_center']).min(dim=-1)[0] / (self['l2_center'].sqrt() + 1e-4))
        angle = torch.sin(2 * angle) - 2
        # Dist Cost
        dist = angle[..., None] * torch.square(self['d_center'] / self['wh_box'])
        dist = 2 - torch.exp(dist[..., 0]) - torch.exp(dist[..., 1])
        # Shape Cost
        d_shape = torch.abs(self['pred_wh'] - self['target_wh'])
        big_shape = torch.maximum(self['pred_wh'], self['target_wh'])
        w_shape = 1 - torch.exp(- d_shape[..., 0] / big_shape[..., 0])
        h_shape = 1 - torch.exp(- d_shape[..., 1] / big_shape[..., 1])
        shape = w_shape ** theta + h_shape ** theta
        return self['iou'] + (dist + shape) / 2
    
    def _MPDIoU(self, mpdiou_hw):
        d1 = (self['target'][..., 0] - self['pred'][..., 0]) ** 2 + (self['target'][..., 1] - self['pred'][..., 1]) ** 2
        d2 = (self['target'][..., 2] - self['pred'][..., 2]) ** 2 + (self['target'][..., 3] - self['pred'][..., 3]) ** 2
        return self['iou'] + d1 / mpdiou_hw + d2 / mpdiou_hw
    
    def _ShapeIoU(self, scale=0.0):
        b1_x1, b1_y1, b1_x2, b1_y2 = self['pred'].chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = self['target'].chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-7
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-7
        
        #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance  
        ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
        hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        c2 = cw ** 2 + ch ** 2 + 1e-7                            # convex diagonal squared
        center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
        center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
        center_distance = hh * center_distance_x + ww * center_distance_y
        distance = center_distance / c2

        #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    
        omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
        return self['iou'] + distance.squeeze() + 0.5 * shape_cost.squeeze()

    def __repr__(self):
        return f'{self.__name__}(iou_mean={self.iou_mean.item():.3f})'

    __name__ = property(lambda self: self.ltype)




def mask_iou(mask1, mask2, eps=1e-7):
    """
    Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
                        product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the
                        product of image width and height.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def kpt_iou(kpt1, kpt2, area, sigma, eps=1e-7):
    """
    Calculate Object Keypoint Similarity (OKS).

    Args:
        kpt1 (torch.Tensor): A tensor of shape (N, 17, 3) representing ground truth keypoints.
        kpt2 (torch.Tensor): A tensor of shape (M, 17, 3) representing predicted keypoints.
        area (torch.Tensor): A tensor of shape (N,) representing areas from ground truth.
        sigma (list): A list containing 17 values representing keypoint scales.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing keypoint similarities.
    """
    d = (kpt1[:, None, :, 0] - kpt2[..., 0]).pow(2) + (kpt1[:, None, :, 1] - kpt2[..., 1]).pow(2)  # (N, M, 17)
    sigma = torch.tensor(sigma, device=kpt1.device, dtype=kpt1.dtype)  # (17, )
    kpt_mask = kpt1[..., 2] != 0  # (N, 17)
    e = d / ((2 * sigma).pow(2) * (area[:, None, None] + eps) * 2)  # from cocoeval
    # e = d / ((area[None, :, None] + eps) * sigma) ** 2 / 2  # from formula
    return ((-e).exp() * kpt_mask[:, None]).sum(-1) / (kpt_mask.sum(-1)[:, None] + eps)


def _get_covariance_matrix(boxes):
    """
    Generating covariance matrix from obbs.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 5) representing rotated bounding boxes, with xywhr format.

    Returns:
        (torch.Tensor): Covariance metrixs corresponding to original rotated bounding boxes.
    """
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor): A tensor of shape (N, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, ) representing obb similarities.
    """
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd
    if CIoU:  # only include the wh aspect ratio part
        w1, h1 = obb1[..., 2:4].split(1, dim=-1)
        w2, h2 = obb2[..., 2:4].split(1, dim=-1)
        v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha  # CIoU
    return iou


def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the prob IoU between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor | np.ndarray): A tensor of shape (N, 5) representing ground truth obbs, with xywhr format.
        obb2 (torch.Tensor | np.ndarray): A tensor of shape (M, 5) representing predicted obbs, with xywhr format.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing obb similarities.
    """
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def smooth_BCE(eps=0.1):
    """
    Computes smoothed positive and negative Binary Cross-Entropy targets.

    This function calculates positive and negative label smoothing BCE targets based on a given epsilon value.
    For implementation details, refer to https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441.

    Args:
        eps (float, optional): The epsilon value for label smoothing. Defaults to 0.1.

    Returns:
        (tuple): A tuple containing the positive and negative label smoothing BCE targets.
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class ConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for predn.
        confusion_matrix_iou (float): The Intersection over Union threshold.
    """
    # 混淆矩阵以及精度和召回率默认iou阈值值0.45
    def __init__(self, nc, conf=0.25, confusion_matrix_iou=0.45, task="detect"):
        """Initialize attributes for the YOLO model."""
        self.confusion_matrix_conf = 0.25 if conf in {None, 0.001} else conf  # apply 0.25 if default val conf is passed
        self.confusion_matrix_iou = confusion_matrix_iou #iou默认0.45
        self.task = task
        self.matrix = np.zeros((nc + 1, nc + 1)) if self.task == "detect" else np.zeros((nc, nc)) #(81, 81)
        self.nc = nc  # number of classes


    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task.

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def update_confusion_matrix_per_image(self, predn, gt_bbox, gt_cls):

        if gt_cls.shape[0] == 0:  # Check if labels is empty
            if predn is not None:
                predn = predn[predn[:, 4] > self.confusion_matrix_conf]
                predn_cls_int = predn[:, 5].int()
                for dc in predn_cls_int:
                    self.matrix[dc, self.nc] += 1  # false positives
            return
        if predn is None:
            gt_cls_int = gt_cls.int()
            for gc in gt_cls_int:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        predn = predn[predn[:, 4] > self.confusion_matrix_conf] #torch.Size([8, 6]) 置信度过滤   0.25
        gt_cls_int = gt_cls.int()
        predn_cls_int = predn[:, 5].int()
        is_obb = predn.shape[1] == 7 and gt_bbox.shape[1] == 5  # with additional `angle` dimension
        bbox_iou = (
            batch_probiou(gt_bbox, torch.cat([predn[:, :4], predn[:, -1:]], dim=-1))
            if is_obb
            else box_iou(gt_bbox, predn[:, :4])
        ) #torch.Size([17, 8])  这里的confusion_matrix_conf 0.25

        iou_index = torch.where(bbox_iou > self.confusion_matrix_iou) #torch.Size([8])  iou过滤   0.45
        if iou_index[0].shape[0]:
            iou_index_value = torch.cat((torch.stack(iou_index, 1), bbox_iou[iou_index[0], iou_index[1]][:, None]), 1).cpu().numpy() #
            if iou_index[0].shape[0] > 1:
                iou_index_value = iou_index_value[iou_index_value[:, 2].argsort()[::-1]]#置信度从大到小重排
                iou_index_value = iou_index_value[np.unique(iou_index_value[:, 1], return_index=True)[1]] #一个真实框对应多个预测框 (8, 3)
                iou_index_value = iou_index_value[iou_index_value[:, 2].argsort()[::-1]]
                iou_index_value = iou_index_value[np.unique(iou_index_value[:, 0], return_index=True)[1]] # (7, 3)
        else:
            iou_index_value = np.zeros((0, 3))

        # have_match = iou_index_value.shape[0] > 0 #-》(7, 3)
        # calculate_gt_index, calculate_predn_index, _ = iou_index_value.transpose().astype(int)
        # for i, gc in enumerate(gt_cls_int):
        #     TF = calculate_gt_index == i #预测是否等于真实
        #     if have_match and sum(TF) == 1:
        #         self.matrix[predn_cls_int[calculate_predn_index[TF]], gc] += 1  # correct
        #     else:
        #         self.matrix[self.nc, gc] += 1  # true background

        # if have_match:
        #     for i, dc in enumerate(predn_cls_int):
        #         if not any(calculate_predn_index == i):
        #             self.matrix[dc, self.nc] += 1  # predicted background


        have_match = iou_index_value.shape[0] > 0  


        calculate_gt_index, calculate_predn_index, _ = iou_index_value.transpose().astype(int)

        # 遍历每个真实框的类别
        for i, gc in enumerate(gt_cls_int):

            TF = calculate_gt_index == i  
            if have_match and sum(TF) == 1:  # 如果存在唯一匹配
                # 将匹配的预测框类别与真实框类别在混淆矩阵中进行更新
                self.matrix[predn_cls_int[calculate_predn_index[TF]], gc] += 1  
            else:
                # 如果没有匹配或有多个匹配，视为真实框的漏检
                self.matrix[self.nc, gc] += 1  

        # 如果有匹配的预测框和真实框
        if have_match:
            # 遍历每个预测框的类别
            for i, pc in enumerate(predn_cls_int):
                # 如果当前预测框未被任何真实框匹配
                if not any(calculate_predn_index == i):
                    # 将预测框视为错误检测，即背景类
                    self.matrix[pc, self.nc] += 1  





    def matrix(self):
        """Returns the confusion matrix."""
        return self.matrix

    def tp_fp(self):
        """Returns true positives and false positives."""
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed predn)
        return (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)  # remove background class if task=detect

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    @plt_settings() #画图
    def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """
        import seaborn  # scope for faster 'import ultralytics'

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (list(names) + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f'{title.lower().replace(" ", "_")}.png'
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """Print the confusion matrix to the console."""
        for i in range(self.nc + 1):
            LOGGER.info(" ".join(map(str, self.matrix[i])))


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


@plt_settings()
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=(), on_plot=None):
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


# Plots ----------------------------------------------------------------------------------------------------------------

# def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=(), on_plot=None):
#     """Plots a precision-recall curve."""
#     fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
#     py = np.stack(py, axis=1)

#     # 改动
#     pr_dict = dict()
#     pr_dict['px'] = px.tolist()


#     if 0 < len(names) < 21:  # display per-class legend if < 21 classes
#         for i, y in enumerate(py.T):
#             ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
#             # 改动2
#             pr_dict[names[i]] = y.tolist()
#     else:
#         ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

#     # 改动3
#     pr_dict['all'] = py.mean(1).tolist()
#     import pandas as pd
#     dataformat = pd.DataFrame(pr_dict)
#     save_csvpath = save_dir.cwd() / (str(save_dir).replace('.png', '.csv')) # 定义csv文件的保存位置
#     dataformat.to_csv(save_csvpath, sep=',')

#     ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
#     ax.set_xlabel("Recall")
#     ax.set_ylabel("Precision")
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#     ax.set_title("Precision-Recall Curve")
#     fig.savefig(save_dir, dpi=250)
#     plt.close(fig)
#     if on_plot:
#         on_plot(save_dir)


@plt_settings() 
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric", on_plot=None):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)

# def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric", on_plot=None):
#     """Plots a metric-confidence curve."""
#     fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

#     # -----------------lwd edit: 将结果保存在csv中--------------- #
#     # 判断是不是绘制F1_curve曲线
#     flag = False
#     if str(save_dir).endswith('F1_curve.png'):
#         flag = True
#         pr_dict = dict()    # lwd edit
#         pr_dict['px'] = px.tolist() # lwd edit
#     # --------------------------------------------------------- #



#     if 0 < len(names) < 21:  # display per-class legend if < 21 classes
#         for i, y in enumerate(py):
#             ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
#             # 改动2
#             if flag:
#                 pr_dict[names[i]] = y.tolist()

#     else:
#         ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)



#     if flag:
#             pr_dict['all'] = y.tolist()
#             import pandas as pd
#             dataformat = pd.DataFrame(pr_dict)
#             save_csvpath = save_dir.cwd() / (str(save_dir).replace('.png', '.csv')) # 定义csv文件的保存位置
#             dataformat.to_csv(save_csvpath, sep=',')
#         # ---------------------------------------------------- #




#     y = smooth(py.mean(0), 0.05)
#     ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#     ax.set_title(f"{ylabel}-Confidence Curve")
#     fig.savefig(save_dir, dpi=250)
#     plt.close(fig)
#     if on_plot:
#         on_plot(save_dir)






def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def ap_per_class(
    tp, conf, pd_cls, gt_cls, plot=False, on_plot=None, save_dir=Path(), names=(), eps=1e-16, prefix=""
):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the predn.
        pd_cls (np.ndarray): Array of predicted classes of the predn.
        gt_cls (np.ndarray): Array of true classes of the predn.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
            fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
            p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
            r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
            f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
            ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
            unique_gt_cls (np.ndarray): An array of unique classes that have data. Shape: (nc,).
            p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
            r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
            f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
            x (np.ndarray): X-axis values for the curves. Shape: (1000,).
            prec_values: Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pd_cls = tp[i], conf[i], pd_cls[i] #按照置信度从大到小排序

    # Find unique classes
    unique_gt_cls, unique_gt_cls_num = np.unique(gt_cls, return_counts=True) #929个目标 71个列别，每个类别的数目
    nc = unique_gt_cls.shape[0]  # number of classes, number of predn

    # Create Precision-Recall curve and compute AP for each class
    x, prec_values = np.linspace(0, 1, 1000), []

    # Average precision, precision and recall curves (71, 10)类别 10档置信度   (71, 1000) 类别 1000
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for unique_gc_index, unique_gc in enumerate(unique_gt_cls):
        cls_TF = pd_cls == unique_gc
        n_p = cls_TF.sum()  # 这一类与多少预测数
        n_l = unique_gt_cls_num[unique_gc_index]  # number of labels  这一类有多少标签数
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[cls_TF]).cumsum(0) #存在错检
        tpc = tp[cls_TF].cumsum(0) #(3737, 10) 最好全是1
        # 按照行的方向（第 0 轴）逐步累加的结果。
        # Recall
        recall = tpc / (n_l + eps)  # recall curve #(3737, 10)
        r_curve[unique_gc_index] = np.interp(-x, -conf[cls_TF], recall[:, 0], left=0)  # negative x, xp because xp decreases   iou阈值是0.5

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p_curve[unique_gc_index] = np.interp(-x, -conf[cls_TF], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[unique_gc_index, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j]) #(71, 10) 71类 每一类10个档位
            if plot and j == 0:
                prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

    prec_values = np.array(prec_values)  # (nc, 1000)

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = [v for k, v in names.items() if k in unique_gt_cls]  # list: only classes that have data
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)

    i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
    tp = (r * unique_gt_cls_num).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_gt_cls.astype(int), p_curve, r_curve, f1_curve, x, prec_values


class Metric(SimpleClass):
    """
    Class for computing evaluation metrics for YOLOv8 model.

    Attributes:
        p (list): Precision for each class. Shape: (nc,).
        r (list): Recall for each class. Shape: (nc,).
        f1 (list): F1 score for each class. Shape: (nc,).
        all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
        ap_class_index (list): Index of class for each AP score. Shape: (nc,).
        nc (int): Number of classes.

    Methods:
        ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
        mp(): Mean precision of all classes. Returns: Float.
        mr(): Mean recall of all classes. Returns: Float.
        map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
        map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
        map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
        mean_results(): Mean of results, returns mp, mr, map50, map.
        class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
        maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
        fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
        update(results): Update metric attributes with new evaluation results.
    """

    def __init__(self) -> None:
        """Initializes a Metric instance for computing evaluation metrics for the YOLOv8 model."""
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50 values per class, or an empty list if not available.
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """
        Returns the Average Precision (AP) at an IoU threshold of 0.5-0.95 for all classes.

        Returns:
            (np.ndarray, list): Array of shape (nc,) with AP50-95 values per class, or an empty list if not available.
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """
        Returns the Mean Precision of all classes.

        Returns:
            (float): The mean precision of all classes.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """
        Returns the Mean Recall of all classes.

        Returns:
            (float): The mean recall of all classes.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.5.

        Returns:
            (float): The mAP at an IoU threshold of 0.5.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """
        Returns the mean Average Precision (mAP) at an IoU threshold of 0.75.

        Returns:
            (float): The mAP at an IoU threshold of 0.75.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map."""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """Class-aware result, return p[i], r[i], ap50[i], ap[i]."""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """MAP of each class."""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        """Model fitness as a weighted combination of metrics."""
        # w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing the following evaluation metrics:
                - p (list): Precision for each class. Shape: (nc,).
                - r (list): Recall for each class. Shape: (nc,).
                - f1 (list): F1 score for each class. Shape: (nc,).
                - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
                - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

        Side Effects:
            Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and `self.ap_class_index` based
            on the values provided in the `results` tuple.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]


class DetMetrics(SimpleClass):
    """
    This class is a utility class for computing detection metrics such as precision, recall, and mean average precision
    (mAP) of an object detection model.

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (tuple of str): A tuple of strings that represents the names of the classes. Defaults to an empty tuple.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (tuple of str): A tuple of strings that represents the names of the classes.
        box (Metric): An instance of the Metric class for storing the results of the detection metrics.
        speed (dict): A dictionary for storing the execution time of different parts of the detection process.

    Methods:
        process(tp, conf, pd_cls, gt_cls): Updates the metric results with the latest batch of predictions.
        keys: Returns a list of keys for accessing the computed detection metrics.
        mean_results: Returns a list of mean values for the computed detection metrics.
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        fitness: Computes the fitness score based on the computed detection metrics.
        ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
        curves: TODO
        curves_results: TODO
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize a DetMetrics instance with a save directory, plot flag, callback function, and class names."""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "detect"

    def process(self, tp, conf, pd_cls, gt_cls):
        """Process predicted results for object detection and update metrics."""
        results = ap_per_class(
            tp, #(21425, 10)
            conf,
            pd_cls,
            gt_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot,
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results


class SegmentMetrics(SimpleClass):
    """
    Calculates and aggregates detection and segmentation metrics over a given set of classes.

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (list): List of class names. Default is an empty list.

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        plot (bool): Whether to save the detection and segmentation plots.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (list): List of class names.
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        seg (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict): Dictionary to store the time taken in different phases of inference.

    Methods:
        process(tp_m, tp_b, conf, pd_cls, gt_cls): Processes metrics over the given set of predictions.
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize a SegmentMetrics instance with a save directory, plot flag, callback function, and class names."""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.seg = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "segment"

    def process(self, tp, tp_m, conf, pd_cls, gt_cls):
        """
        Processes the detection and segmentation metrics over the given set of predictions.

        Args:
            tp (list): List of True Positive boxes.
            tp_m (list): List of True Positive masks.
            conf (list): List of confidence scores.
            pd_cls (list): List of predicted classes.
            gt_cls (list): List of target classes.
        """

        results_mask = ap_per_class(
            tp_m,
            conf,
            pd_cls,
            gt_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Mask",
        )[2:]
        self.seg.nc = len(self.names)
        self.seg.update(results_mask)
        results_box = ap_per_class(
            tp,
            conf,
            pd_cls,
            gt_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self): #分割的指标
        """Returns a list of keys for accessing metrics."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
        ]

    def mean_results(self):
        """Return the mean metrics for bounding box and segmentation results."""
        return self.box.mean_results() + self.seg.mean_results()

    def class_result(self, i):
        """Returns classification results for a specified class index."""
        return self.box.class_result(i) + self.seg.class_result(i)

    @property
    def maps(self):
        """Returns mAP scores for object detection and semantic segmentation models."""
        return self.box.maps + self.seg.maps

    @property
    def fitness(self):
        """Get the fitness score for both segmentation and bounding box models."""
        return self.seg.fitness() + self.box.fitness()
        # return self.seg.fitness()
    
    @property
    def ap_class_index(self):
        """Boxes and masks have the same ap_class_index."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns results of object detection model for evaluation."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(M)",
            "F1-Confidence(M)",
            "Precision-Confidence(M)",
            "Recall-Confidence(M)",
        ]

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results + self.seg.curves_results


class PoseMetrics(SegmentMetrics):
    """
    Calculates and aggregates detection and pose metrics over a given set of classes.

    Args:
        save_dir (Path): Path to the directory where the output plots should be saved. Default is the current directory.
        plot (bool): Whether to save the detection and segmentation plots. Default is False.
        on_plot (func): An optional callback to pass plots path and data when they are rendered. Defaults to None.
        names (list): List of class names. Default is an empty list.

    Attributes:
        save_dir (Path): Path to the directory where the output plots should be saved.
        plot (bool): Whether to save the detection and segmentation plots.
        on_plot (func): An optional callback to pass plots path and data when they are rendered.
        names (list): List of class names.
        box (Metric): An instance of the Metric class to calculate box detection metrics.
        pose (Metric): An instance of the Metric class to calculate mask segmentation metrics.
        speed (dict): Dictionary to store the time taken in different phases of inference.

    Methods:
        process(tp_m, tp_b, conf, pd_cls, gt_cls): Processes metrics over the given set of predictions.
        mean_results(): Returns the mean of the detection and segmentation metrics over all the classes.
        class_result(i): Returns the detection and segmentation metrics of class `i`.
        maps: Returns the mean Average Precision (mAP) scores for IoU thresholds ranging from 0.50 to 0.95.
        fitness: Returns the fitness scores, which are a single weighted combination of metrics.
        ap_class_index: Returns the list of indices of classes used to compute Average Precision (AP).
        results_dict: Returns the dictionary containing all the detection and segmentation metrics and fitness score.
    """

    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize the PoseMetrics class with directory path, class names, and plotting options."""
        super().__init__(save_dir, plot, names)
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.pose = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "pose"

    def process(self, tp, tp_p, conf, pd_cls, gt_cls):
        """
        Processes the detection and pose metrics over the given set of predictions.

        Args:
            tp (list): List of True Positive boxes.
            tp_p (list): List of True Positive keypoints.
            conf (list): List of confidence scores.
            pd_cls (list): List of predicted classes.
            gt_cls (list): List of target classes.
        """

        results_pose = ap_per_class(
            tp_p,
            conf,
            pd_cls,
            gt_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Pose",
        )[2:]
        self.pose.nc = len(self.names)
        self.pose.update(results_pose)
        results_box = ap_per_class(
            tp,
            conf,
            pd_cls,
            gt_cls,
            plot=self.plot,
            on_plot=self.on_plot,
            save_dir=self.save_dir,
            names=self.names,
            prefix="Box",
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results_box)

    @property
    def keys(self):
        """Returns list of evaluation metric keys."""
        return [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(P)",
            "metrics/recall(P)",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
        ]

    def mean_results(self):
        """Return the mean results of box and pose."""
        return self.box.mean_results() + self.pose.mean_results()

    def class_result(self, i):
        """Return the class-wise detection results for a specific class i."""
        return self.box.class_result(i) + self.pose.class_result(i)

    @property
    def maps(self):
        """Returns the mean average precision (mAP) per class for both box and pose predn."""
        return self.box.maps + self.pose.maps

    @property
    def fitness(self):
        """Computes classification metrics and speed using the `targets` and `pred` inputs."""
        return self.pose.fitness() + self.box.fitness()

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return [
            "Precision-Recall(B)",
            "F1-Confidence(B)",
            "Precision-Confidence(B)",
            "Recall-Confidence(B)",
            "Precision-Recall(P)",
            "F1-Confidence(P)",
            "Precision-Confidence(P)",
            "Recall-Confidence(P)",
        ]

    @property
    def curves_results(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return self.box.curves_results + self.pose.curves_results


class ClassifyMetrics(SimpleClass):
    """
    Class for computing classification metrics including top-1 and top-5 accuracy.

    Attributes:
        top1 (float): The top-1 accuracy.
        top5 (float): The top-5 accuracy.
        speed (Dict[str, float]): A dictionary containing the time taken for each step in the pipeline.
        fitness (float): The fitness of the model, which is equal to top-5 accuracy.
        results_dict (Dict[str, Union[float, str]]): A dictionary containing the classification metrics and fitness.
        keys (List[str]): A list of keys for the results_dict.

    Methods:
        process(targets, pred): Processes the targets and predictions to compute classification metrics.
    """

    def __init__(self) -> None:
        """Initialize a ClassifyMetrics instance."""
        self.top1 = 0
        self.top5 = 0
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "classify"

    def process(self, targets, pred):
        """Target classes and predicted classes."""
        pred, targets = torch.cat(pred), torch.cat(targets)
        correct = (targets[:, None] == pred).float()
        acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
        self.top1, self.top5 = acc.mean(0).tolist()

    @property
    def fitness(self):
        """Returns mean of top-1 and top-5 accuracies as fitness score."""
        return (self.top1 + self.top5) / 2

    @property
    def results_dict(self):
        """Returns a dictionary with model's performance metrics and fitness score."""
        return dict(zip(self.keys + ["fitness"], [self.top1, self.top5, self.fitness]))

    @property
    def keys(self):
        """Returns a list of keys for the results_dict property."""
        return ["metrics/accuracy_top1", "metrics/accuracy_top5"]

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []


class OBBMetrics(SimpleClass):
    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names=()) -> None:
        """Initialize an OBBMetrics instance with directory, plotting, callback, and class names."""
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

    def process(self, tp, conf, pd_cls, gt_cls):
        """Process predicted results for object detection and update metrics."""
        results = ap_per_class(
            tp,
            conf,
            pd_cls,
            gt_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot,
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        """Returns a list of keys for accessing specific metrics."""
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self):
        """Calculate mean of detected objects & return precision, recall, mAP50, and mAP50-95."""
        return self.box.mean_results()

    def class_result(self, i):
        """Return the result of evaluating the performance of an object detection model on a specific class."""
        return self.box.class_result(i)

    @property
    def maps(self):
        """Returns mean Average Precision (mAP) scores per class."""
        return self.box.maps

    @property
    def fitness(self):
        """Returns the fitness of box object."""
        return self.box.fitness()

    @property
    def ap_class_index(self):
        """Returns the average precision index per class."""
        return self.box.ap_class_index

    @property
    def results_dict(self):
        """Returns dictionary of computed performance metrics and statistics."""
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []

    @property
    def curves_results(self):
        """Returns a list of curves for accessing specific metrics curves."""
        return []
