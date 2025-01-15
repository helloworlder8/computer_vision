# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher


class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class. This class calculates and returns the different loss components for the
    DETR object detection model. It computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary
    avg_loss_items.

    Attributes:
        nc (int): The number of classes.
        loss_gain (dict): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary avg_loss_items.
        use_fl (bool): Use FocalLoss or not.
        use_vfl (bool): Use VarifocalLoss or not.
        use_uni_match (bool): Whether to use a fixed layer to assign labels for the auxiliary branch.
        uni_match_ind (int): The fixed indices of a layer to use if `use_uni_match` is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss or None): Focal Loss object if `use_fl` is True, otherwise None.
        vfl (VarifocalLoss or None): Varifocal Loss object if `use_vfl` is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self, nc=80, loss_gain=None, aux_loss=True, use_fl=True, use_vfl=True, use_uni_match=False, uni_match_ind=0
    ):
        """
        DETR loss function.

        Args:
            nc (int): The number of classes.
            loss_gain (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_vfl (bool): Use VarifocalLoss or not.
            use_uni_match (bool): Whether to use a fixed layer to assign labels for auxiliary branch.
            uni_match_ind (int): The fixed indices of a layer.
        """
        super().__init__()
        self.nc = nc
        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss

        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(self, pd_scores, targets, gt_scores, num_gts, postfix=""):
        """Computes the classification loss based on predictions, target values, and ground truth scores."""
        # Logits: [b, query, nc], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, num_top_k = pd_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, nc)
        one_hot = torch.zeros((bs, num_top_k, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, num_top_k, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pd_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pd_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / num_top_k
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pd_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(self, pd_bboxes, gt_bboxes, postfix=""):
        """Calculates and returns the bounding box loss and GIoU loss for the predicted and ground truth bounding
        boxes.
        """
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pd_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pd_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes) #平均每一个框的损失
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    # This function is for future RT-DETR Segment models
    # def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # This function is for future RT-DETR Segment models
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts

    def _get_loss_aux(
        self,
        pd_bboxes,
        pd_scores,
        gt_bboxes,
        gt_cls,
        bboxs_each_img,
        match_indices=None,
        postfix="",
        masks=None,
        gt_mask=None,
    ):
        """Get auxiliary avg_loss_items."""
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pd_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pd_bboxes[self.uni_match_ind],
                pd_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                bboxs_each_img,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pd_bboxes, pd_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                bboxs_each_img,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    @staticmethod
    def _get_index(match_indices):
        """Returns batch indices, source indices, and destination indices from provided match indices."""

        batch_idx = []


        for i, (src, _) in enumerate(match_indices):

            img_idx = torch.full_like(src, i)

            batch_idx.append(img_idx)

        # 最后将所有的 img_idx 合并成一个张量
        img_idx = torch.cat(batch_idx) #torch.Size([30]) 解析生成 batch_idx
        pd_idx = torch.cat([src for (src, _) in match_indices])
        gt_idx = torch.cat([dst for (_, dst) in match_indices])
        return (img_idx, pd_idx), gt_idx

    def _get_assigned_bboxes(self, pd_bboxes, gt_bboxes, match_indices):
        """Assigns predicted bounding boxes to ground truth bounding boxes based on the match indices."""
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pd_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pd_bboxes,
        pd_scores,
        gt_bboxes,
        gt_cls,
        bboxs_each_img,
        masks=None,
        gt_mask=None,
        postfix="",
        match_indices=None,
    ):
        """Get avg_loss_items."""
        if match_indices is None:
            match_indices = self.matcher(
                pd_bboxes, pd_scores, gt_bboxes, gt_cls, bboxs_each_img, masks=masks, gt_mask=gt_mask
            )
        # match_indices最终的预测框索引和真实框索引
        idx, gt_idx = self._get_index(match_indices) #就是一个数据格式的转换
        pd_bboxes, gt_bboxes = pd_bboxes[idx], gt_bboxes[gt_idx] #匹配好后的bboxes

        bs, num_top_k = pd_scores.shape[:2]
        targets = torch.full((bs, num_top_k), self.nc, device=pd_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, num_top_k], device=pd_scores.device) #torch.Size([4, 300])
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pd_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        loss = {}
        loss.update(self._get_loss_class(pd_scores, targets, gt_scores, len(gt_bboxes), postfix))
        loss.update(self._get_loss_bbox(pd_bboxes, gt_bboxes, postfix))
        # if masks is not None and gt_mask is not None:
        #     loss.update(self._get_loss_mask(masks, gt_mask, match_indices, postfix))
        return loss

    def forward(self, pd_bboxes, pd_scores, gt, postfix="", **kwargs):
        """
        Args:
            pd_bboxes (torch.Tensor): [l, b, query, 4]
            pd_scores (torch.Tensor): [l, b, query, nc]
            gt (dict): A dict includes:
                gt_cls (torch.Tensor) with shape [num_gts, ],
                gt_bboxes (torch.Tensor): [num_gts, 4],
                bboxs_each_img (List(int)): a list of gt size length includes the number of gts of each image.
            postfix (str): postfix of loss name.
        """
        self.device = pd_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, bboxs_each_img = gt["cls"], gt["bboxes"], gt["bboxs_each_img"]
        # {'loss_class': torch.Size([]) tensor(1.2887, device='cuda:0', grad_fn=<MulBackward0>), 'loss_bbox': torch.Size([]) tensor(0.3082, device='cuda:0', grad_fn=<SqueezeBackward0>), 'loss_giou': torch.Size([]) tensor(0.3919, device='cuda:0', grad_fn=<SqueezeBackward0>)}
        total_loss = self._get_loss(
            pd_bboxes[-1], pd_scores[-1], gt_bboxes, gt_cls, bboxs_each_img, postfix=postfix, match_indices=match_indices
        )

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    pd_bboxes[:-1], pd_scores[:-1], gt_bboxes, gt_cls, bboxs_each_img, match_indices, postfix
                )
            )

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def forward(self, pd, gt, dn_bboxes=None, dn_cls=None, dn_meta=None):
        """
        Forward pass to compute the detection loss.

        Args:
            pd (tuple): Predicted bounding boxes and scores.
            gt (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. Default is None.
            dn_cls (torch.Tensor, optional): Denoising scores. Default is None.
            dn_meta (dict, optional): Metadata for denoising. Default is None.

        Returns:
            (dict): Dictionary containing the total loss and, if applicable, the denoising loss.
        """
        pd_bboxes, pd_scores = pd
        total_loss = super().forward(pd_bboxes, pd_scores, gt)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, num_dn_group = dn_meta["dn_pos_idx"], dn_meta["num_dn_group"]
            assert len(gt["bboxs_each_img"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, num_dn_group, gt["bboxs_each_img"])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_cls, gt, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, num_dn_group, bboxs_each_img):
        """
        Get the match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            num_dn_group (int): Number of denoising groups.
            bboxs_each_img (List[int]): List of integers representing the number of ground truths for each image.

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *bboxs_each_img[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(bboxs_each_img):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(num_dn_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), "Expected the same length, "
                f"but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices
