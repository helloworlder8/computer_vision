# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, args=None, dataloader=None, save_dir=None, pbar=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(args, dataloader, save_dir, pbar, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect" #有点强调
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        if self.args.save_hybrid:
            LOGGER.warning(
                "WARNING ⚠️ 'save_hybrid=True' will append ground truth to predictions for autolabelling.\n"
                "WARNING ⚠️ 'save_hybrid=True' will cause incorrect mAP.\n"
            )

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        # 获取 "ch" 键的值，如果不存在则返回 None
        ch = self.data_dict.get("ch")

        # 检查 "ch" 的值是否为 1，且不为 None
        if ch == 1:
            # 兼容单通道数据
            batch['img'] = batch['img'][:, :1, :, :]
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data_dict.get(self.args.split, "")  # validation path
        # self.is_coco = (
        #     isinstance(val, str)
        #     and "coco" in val
        #     and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        # )  # is COCO
        # self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        # self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))
        # self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats_dict = dict(tp=[], conf=[], pd_cls=[], gt_cls=[], target_unique_cls=[])

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf, #0.001
            self.args.NMS_IoU,  #0.7
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, pred_index, batch): #batch是一批真实数据
        """Prepares a batch of images and annotations for validation."""
        TF = batch["batch_idx"] == pred_index #
        bbox = batch["bboxes"][TF] #torch.Size([17, 4])
        cls = batch["cls"][TF].squeeze(-1) #torch.Size([17])
        ori_shape = batch["ori_shape"][pred_index] #[329, 635]
        imgsz = batch["img"].shape[2:] #[218, 640]
        ratio_pad = batch["ratio_pad"][pred_index] #[[1.0, 1.0], [16, 19]]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"bbox": bbox, "cls": cls, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, target_dict):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            target_dict["imgsz"], predn[:, :4], target_dict["ori_shape"], ratio_pad=target_dict["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for pred_index, pred in enumerate(preds):
            self.seen += 1
            num_pred = len(pred) #228
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pd_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(num_pred, self.niou, dtype=torch.bool, device=self.device), #torch.Size([228, 10])
            )
            target_dict = self._prepare_batch(pred_index, batch) #17个目标
            gt_cls, target_bbox = target_dict.pop("cls"), target_dict.pop("bbox")
            num_target = len(gt_cls) #-》17
            stat["gt_cls"] = gt_cls #真是的有多少个框
            stat["target_unique_cls"] = gt_cls.unique() #框总共多少类
            if num_pred == 0:
                if num_target:
                    for k in self.stats_dict.keys():
                        self.stats_dict[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(predn=None, gt_bboxes=target_bbox, gt_cls=gt_cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, target_dict) #尺度缩放
            stat["conf"] = predn[:, 4] #预测结果的置信度
            stat["pd_cls"] = predn[:, 5] #预测结果的类别

            # Evaluate
            if num_target:
                stat["tp"] = self._process_batch(predn, target_bbox, gt_cls) #torch.Size([228, 6]) torch.Size([17, 4]) torch.Size([17]) torch.Size([228, 10])
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, target_bbox, gt_cls)
            for k in self.stats_dict.keys():
                self.stats_dict[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][pred_index])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    target_dict["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][pred_index]).stem}.txt',
                )

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats_dict.items()}  # (21425, 10)
        self.nt_per_class = np.bincount(stats["gt_cls"].astype(int), minlength=self.nc)#929个框的80个类的统计结果
        self.nt_per_image = np.bincount(stats["target_unique_cls"].astype(int), minlength=self.nc) #(80,) 这玩意也没用到啊
        stats.pop("target_unique_cls", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        return self.metrics.results_dict

    def print_results(self):  #打印指标
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats_dict):
            detect_metrics_file = os.path.join(self.save_dir, 'DetectMetrics.txt')
            with open(detect_metrics_file, 'a') as f:
                for i, c in enumerate(self.metrics.ap_class_index):
                    log_message = pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                    LOGGER.info(log_message)
                    f.write(log_message + '\n')

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, predn, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            predn (torch.Tensor): Tensor of shape (N, 6) representing predn where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, predn[:, :4]) #torch.Size([17, 228]) 17是真实 228是预测
        return self.create_pd_iouv_matrix(predn[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, batch_size=None, mode="val"):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, self.data_dict, batch_size, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_str, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_str, batch_size=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO predn to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])]
                    + (1 if self.is_lvis else 0),  # index starts from 1 if it's lvis
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results
                # update mAP50-95 and mAP50
                stats[self.metrics.results_dict[-1]], stats[self.metrics.results_dict[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats
