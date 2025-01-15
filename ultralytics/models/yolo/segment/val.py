# Ultralytics YOLO 🚀, AGPL-3.0 license

from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, NUM_THREADS, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import SegmentMetrics, box_iou, mask_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class SegmentationValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a segmentation model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import SegmentationValidator

        args = dict(model='yolov8n-seg.pt', data='coco8-seg.yaml')
        validator = SegmentationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, args=None, dataloader=None, save_dir=None, pbar=None, _callbacks=None):
        """Initialize SegmentationValidator and set task to 'segment', metrics to SegmentMetrics."""
        super().__init__(args, dataloader, save_dir, pbar, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = "segment"
        self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch = super().preprocess(batch)
        batch["masks"] = batch["masks"].to(self.device).float()
        return batch

    def init_metrics(self, model):
        """Initialize metrics and select mask processing function based on save_json flag."""
        super().init_metrics(model)
        self.plot_masks = []
        if self.args.save_json:
            check_requirements("pycocotools>=2.0.6")
        # more accurate vs faster
        self.process = ops.process_mask_native if self.args.save_json or self.args.save_txt else ops.process_mask
        self.stats_dict = dict(tp_m=[], tp=[], conf=[], pd_cls=[], tgt_cls=[], tgt_unique_cls=[])

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Mask(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """Post-processes YOLO predictions and returns output predn with proto."""
        p = ops.non_max_suppression( #116 = 4+80+32  38 = 4+1+32
            preds[0],
            self.args.conf, #评估的置信度是0.001
            self.args.NMS_Threshold, #nms阈值是0.7
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )
        proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
        return p, proto

    def _get_tgt_dict_per_image(self, i, batch): #从属性到批
        """Prepares a batch for training or inference by processing images and targets."""
        tgt_dict = super()._get_tgt_dict_per_image(i, batch)
        mask_ind = [i] if self.args.overlap_mask else batch["img_idx"] == i
        tgt_dict["masks"] = batch["masks"][mask_ind]
        return tgt_dict

    def _generate_native_space_predn(self, pred_i, tgt_dict_i, pred_proto_i): #pred_i, tgt_dict_i, pred_proto_i
        """Prepares a batch for training or inference by processing images and targets."""
        predn_i = super()._generate_native_space_predn(pred_i, tgt_dict_i)
        pd_seg_masks_i = self.process(pred_proto_i, pred_i[:, 6:], pred_i[:, :4], shape=tgt_dict_i["imgsz"])
        return predn_i, pd_seg_masks_i

    def update_metrics_per_batch(self, preds, batch):
        """Metrics."""
        for i, (pred_i, pred_proto_i) in enumerate(zip(preds[0], preds[1])):
            self.seen += 1
            num_pre = len(pred_i)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pd_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(num_pre, self.niou, dtype=torch.bool, device=self.device),
                tp_m=torch.zeros(num_pre, self.niou, dtype=torch.bool, device=self.device),
            )
            tgt_dict_i = self._get_tgt_dict_per_image(i, batch)
            tgt_cls_i, tgt_bbox_i = tgt_dict_i.pop("cls"), tgt_dict_i.pop("bbox")
            num_tgt = len(tgt_cls_i)
            stat["tgt_cls"] = tgt_cls_i
            stat["tgt_unique_cls"] = tgt_cls_i.unique()
            if num_pre == 0:
                if num_tgt:
                    for k in self.stats_dict.keys():
                        self.stats_dict[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.update_confusion_matrix_per_image(predn=None, gt_bbox=tgt_bbox_i, gt_cls=tgt_cls_i)
                continue

            # Masks
            tgt_masks = tgt_dict_i.pop("masks")
            # Predictions
            if self.args.single_cls:
                pred_i[:, 5] = 0
            predn_i, pd_seg_masks_i = self._generate_native_space_predn(pred_i, tgt_dict_i, pred_proto_i)
            stat["conf"] = predn_i[:, 4]
            stat["pd_cls"] = predn_i[:, 5]

            # Evaluate
            if num_tgt:
                stat["tp"] = self._generate_per_image_tp(predn_i, tgt_bbox_i, tgt_cls_i)
                stat["tp_m"] = self._generate_per_image_tp(
                    predn_i, tgt_bbox_i, tgt_cls_i, pd_seg_masks_i, tgt_masks, self.args.overlap_mask, masks=True
                )
                if self.args.plots:
                    self.confusion_matrix.update_confusion_matrix_per_image(predn_i, tgt_bbox_i, tgt_cls_i)

            for k in self.stats_dict.keys():
                self.stats_dict[k].append(stat[k])

            pd_seg_masks_i = torch.as_tensor(pd_seg_masks_i, dtype=torch.uint8)
            if self.args.plots and self.batch_index < 3:
                self.plot_masks.append(pd_seg_masks_i[:15].cpu())  # filter top 15 to plot

            # Save
            if self.args.save_json:
                self.pred_to_json(
                    predn_i,
                    batch["im_file"][i],
                    ops.scale_image(
                        pd_seg_masks_i.permute(1, 2, 0).contiguous().cpu().numpy(),
                        tgt_dict_i["ori_shape"],
                        ratio_pad=batch["ratio_pad"][i],
                    ),
                )
            if self.args.save_txt:
                self.save_one_txt(
                    predn_i,
                    pd_seg_masks_i,
                    self.args.save_conf,
                    tgt_dict_i["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][i]).stem}.txt',
                )

    def finalize_add_metrics(self, *args, **kwargs):
        """Sets speed and confusion matrix for evaluation metrics."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        
        
    # predn_i, tgt_bbox_i, tgt_cls_i
    def _generate_per_image_tp(self, predn_i, tgt_bbox_i, tgt_cls_i, pd_seg_matrix=None, tgt_masks=None, overlap=False, masks=False):
        """
        Example:
            ```python
            predn = torch.tensor([[25, 30, 200, 300, 0.8, 1], [50, 60, 180, 290, 0.75, 0]])
            gt_bboxes = torch.tensor([[24, 29, 199, 299], [55, 65, 185, 295]])
            tgt_cls_i = torch.tensor([1, 0])
            correct_preds = validator._generate_per_image_tp(predn, gt_bboxes, tgt_cls_i)
            ```
        """
        if masks:
            if overlap:
                num_tgt = len(tgt_cls_i)
                index = torch.arange(num_tgt, device=tgt_masks.device).view(num_tgt, 1, 1) + 1
                tgt_masks = tgt_masks.repeat(num_tgt, 1, 1)  # shape(1,640,640) -> (n,640,640)
                tgt_masks = torch.where(tgt_masks == index, 1.0, 0.0)
            if tgt_masks.shape[1:] != pd_seg_matrix.shape[1:]:
                tgt_masks = F.interpolate(tgt_masks[None], pd_seg_matrix.shape[1:], mode="bilinear", align_corners=False)[0]
                tgt_masks = tgt_masks.gt_(0.5)
            iou = mask_iou(tgt_masks.view(tgt_masks.shape[0], -1), pd_seg_matrix.view(pd_seg_matrix.shape[0], -1))
        else:  # boxes
            iou = box_iou(tgt_bbox_i, predn_i[:, :4]) #这个iou是真实和预测的交集 torch.Size([8, 300])
        # 可以理解预测掩码的数据标注格式在一个1X640x640的矩阵上有各个类别的信息
        return self.create_tp_iouv_matrix(predn_i[:, 5], tgt_cls_i, iou)

    def plot_val_samples(self, batch, ni):
        """Plots validation samples with bounding box labels."""
        plot_images(
            batch["img"],
            batch["img_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots batch predictions with masks and bounding boxes."""
        plot_images(
            batch["img"],
            *output_to_target(preds[0], max_det=15),  # not set to self.args.max_det due to slow plotting speed
            torch.cat(self.plot_masks, dim=0) if len(self.plot_masks) else self.plot_masks,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred
        self.plot_masks.clear()

    def save_one_txt(self, predn, pd_seg_matrix, save_conf, shape, file):
        """Save YOLO predn to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
            masks=pd_seg_matrix,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename, pd_seg_matrix):
        """
        Save one JSON result.

        Examples:
             >>> result = {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
        """
        from pycocotools.mask import encode  # noqa

        def single_encode(x):
            """Encode predicted masks as RLE and append results to jdict."""
            rle = encode(np.asarray(x[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        pd_seg_matrix = np.transpose(pd_seg_matrix, (2, 0, 1))
        with ThreadPool(NUM_THREADS) as pool:
            rles = pool.map(single_encode, pd_seg_matrix)
        for i, (p, b) in enumerate(zip(predn.tolist(), box.tolist())):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                    "segmentation": rles[i],
                }
            )

    def eval_json(self, stats):
        """Return COCO-style object detection evaluation metrics."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/instances_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements("pycocotools>=2.0.6")
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "segm")]):
                    if self.is_coco:
                        eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # img to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    stats[self.metrics.results_dict[idx + 1]], stats[self.metrics.results_dict[idx]] = eval.stats[
                        :2
                    ]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f"pycocotools unable to run: {e}")
        return stats
