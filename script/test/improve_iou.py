import numpy as np
import torch, math
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
#           91 4   91 4
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, WIoU=False, Focal=False, pow=1, gamma=0.5, scale=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1) #list(tensor(list))
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1) #91 1
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1) 
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps  #宽 高
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps 



    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    if scale:
        self = WIoU_Scale(1 - (inter / union))

    # IoU
    # iou = inter / union # ori iou
    iou = torch.pow(inter/(union + eps), pow) # pow iou

    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU:
        convex_wid = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        convex_hei = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or WIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            convex_area_squared = (convex_wid ** 2 + convex_hei ** 2) ** pow + eps  # convex diagonal squared外接面积平方
            center_dist_wid = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps #中心点距离之差 宽
            center_dist_hei = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps #中心点距离之差 高
            center_dist_squared = torch.pow(center_dist_wid ** 2 + center_dist_hei ** 2, pow)  # 中心距离平方
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorconvex_hei/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    param_ciou = v / (v - iou + (1 + eps))
                if Focal:
                    return iou - (center_dist_squared / convex_area_squared + torch.pow(v * param_ciou + eps, pow)), torch.pow(inter/(union + eps), gamma)  # Focal_CIoU
                else:


                    """ CIoU """
                    return iou - (center_dist_squared / convex_area_squared + torch.pow(v * param_ciou + eps, pow))  # CIoU
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2 #torch.Size([53121, 1])宽度差的平方
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2 #torch.Size([53121, 1])高度差的平方
                convex_wid2 = torch.pow(convex_wid ** 2 + eps, pow)
                convex_hei2 = torch.pow(convex_hei ** 2 + eps, pow)
                if Focal:
                    return iou - (center_dist_squared / convex_area_squared + rho_w2 / convex_wid2 + rho_h2 / convex_hei2), torch.pow(inter/(union + eps), gamma) # Focal_EIou
                else:
                    return iou - (center_dist_squared / convex_area_squared + rho_w2 / convex_wid2 + rho_h2 / convex_hei2) # EIou
            elif SIoU:
                # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
                center_dist = torch.pow(center_dist_wid ** 2 + center_dist_hei ** 2, 0.5) #计算中心点距离
                sin_pow_1 = torch.abs(center_dist_wid) / center_dist #宽比距离
                sin_pow_2 = torch.abs(center_dist_hei) / center_dist #高比距离
                threshold = 2 ** 0.5 / 2
                sin_pow = torch.where(sin_pow_1 > threshold, sin_pow_2, sin_pow_1) #宽高正弦值选择

                angle_cost = torch.cos(torch.arcsin(sin_pow) * 2 - math.pi / 2) #角度成本


                rho_x = (center_dist_wid / convex_wid) ** 2 #距离成本
                rho_y = (center_dist_hei / convex_hei) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)


                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2) #形状成本
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)

                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                if Focal:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, pow), torch.pow(inter/(union + eps), gamma) # Focal_SIou
                else:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, pow) # SIou
            
            elif WIoU:
                if Focal:
                    raise RuntimeError("WIoU do not support Focal.")
                elif scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(self), (1 - iou) * torch.exp((center_dist_squared / convex_area_squared)), iou # WIoU https://arxiv.org/abs/2301.10051
                else:
                    return iou, torch.exp((center_dist_squared / convex_area_squared)) # WIoU v1
            if Focal:
                return iou - center_dist_squared / convex_area_squared, torch.pow(inter/(union + eps), gamma)  # Focal_DIoU
            else:


                """ DIoU """
                return iou - center_dist_squared / convex_area_squared  # DIoU
        c_area = convex_wid * convex_hei + eps  # convex area
        if Focal:
            return iou - torch.pow((c_area - union) / c_area + eps, pow), torch.pow(inter/(union + eps), gamma)  # Focal_GIoU https://arxiv.org/pdf/1902.09630.pdf
        else:
            return iou - torch.pow((c_area - union) / c_area + eps, pow)  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    if Focal:
        return iou, torch.pow(inter/(union + eps), gamma)  # Focal_IoU
    else:
        return iou  # IoU



def test_bbox_iou():
    # 创建两个包围框张量
    with open('runs/debug_param/box1.pkl', 'rb') as f:
        box1 = pickle.load(f)
    with open('runs/debug_param/box2.pkl', 'rb') as f:
        box2 = pickle.load(f)

    # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True) #x1 y1 x2 y2用的两顶点坐标
    # 计算 IoU
    iou = bbox_iou(box1, box2, xywh=False, SIoU=True)

    print("IoU values:")
    print(iou)

if __name__ == "__main__":
    test_bbox_iou()
