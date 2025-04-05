# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, box_iou, kpt_iou
from ultralytics.utils.plotting import output_to_target, plot_images


class PoseValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a pose model.

    Example:
        ```python
        from ultralytics.models.yolo.pose import PoseValidator

        args = dict(model='yolov8n-pose.pt', data='coco8-pose.yaml')
        validator = PoseValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize a 'PoseValidator' object with custom parameters and assigned attributes."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch):
        """Preprocesses the batch by converting the 'keypoints' data into a float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        return batch

    def get_desc(self):
        """Returns description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def postprocess(self, preds):
        """Apply non-maximum suppression and return predn with high confidence scores."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.NMS_Threshold,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
            nc=self.nc,
        )

    def init_metrics(self, model):
        """Initiate pose estimation metrics for YOLO model."""
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt
        self.stats_dict = dict(tp_p=[], tp=[], conf=[], pd_cls=[], gt_cls=[], tgt_unique_cls=[])

    def _get_tgt_dict_per_image(self, si, batch):
        """Prepares a batch for processing by converting keypoints to float and moving to device."""
        tgt_dict_i = super()._get_tgt_dict_per_image(si, batch)
        kpts = batch["keypoints"][batch["img_idx"] == si]
        h, w = tgt_dict_i["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        kpts = ops.scale_coords(tgt_dict_i["imgsz"], kpts, tgt_dict_i["ori_shape"], ratio_pad=tgt_dict_i["ratio_pad"])
        tgt_dict_i["kpts"] = kpts
        return tgt_dict_i

    def _generate_native_space_predn(self, pred, tgt_dict_i):
        """Prepares and scales keypoints in a batch for pose processing."""
        predn = super()._generate_native_space_predn(pred, tgt_dict_i)
        nk = tgt_dict_i["kpts"].shape[1]
        pred_kpts = predn[:, 6:].view(len(predn), nk, -1)
        ops.scale_coords(tgt_dict_i["imgsz"], pred_kpts, tgt_dict_i["ori_shape"], ratio_pad=tgt_dict_i["ratio_pad"])
        return predn, pred_kpts

    def update_metrics_per_batch(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pd_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                tp_p=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            tgt_dict_i = self._get_tgt_dict_per_image(si, batch)
            cls, bbox = tgt_dict_i.pop("cls"), tgt_dict_i.pop("bbox")
            nl = len(cls)
            stat["tgt_cls"] = cls
            stat["tgt_unique_cls"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats_dict.keys():
                        self.stats_dict[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.update_confusion_matrix_per_image(predn=None, gt_bbox=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn, pred_kpts = self._generate_native_space_predn(pred, tgt_dict_i)
            stat["conf"] = predn[:, 4]
            stat["pd_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._generate_per_image_tp(predn, bbox, cls)
                stat["tp_p"] = self._generate_per_image_tp(predn, bbox, cls, pred_kpts, tgt_dict_i["kpts"])
                if self.args.plots:
                    self.confusion_matrix.update_confusion_matrix_per_image(predn, bbox, cls)

            for k in self.stats_dict.keys():
                self.stats_dict[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    pred_kpts,
                    self.args.save_conf,
                    tgt_dict_i["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )

    def _generate_per_image_tp(self, predn, gt_bboxes, gt_cls, pred_kpts=None, gt_kpts=None):
        """
        Return correct prediction matrix by computing Intersection over Union (IoU) between predn and ground truth.

        Args:
            predn (torch.Tensor): Tensor with shape (N, 6) representing detection boxes and scores, where each
                detection is of the format (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor with shape (M, 4) representing ground truth bounding boxes, where each
                box is of the format (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor with shape (M,) representing ground truth class indices.
            pred_kpts (torch.Tensor | None): Optional tensor with shape (N, 51) representing predicted keypoints, where
                51 corresponds to 17 keypoints each having 3 values.
            gt_kpts (torch.Tensor | None): Optional tensor with shape (N, 51) representing ground truth keypoints.

        Returns:
            torch.Tensor: A tensor with shape (N, 10) representing the correct prediction matrix for 10 IoU levels,
                where N is the number of predn.

        Example:
            ```python
            predn = torch.rand(100, 6)  # 100 predictions: (x1, y1, x2, y2, conf, class)
            gt_bboxes = torch.rand(50, 4)  # 50 ground truth boxes: (x1, y1, x2, y2)
            gt_cls = torch.randint(0, 2, (50,))  # 50 ground truth class indices
            pred_kpts = torch.rand(100, 51)  # 100 predicted keypoints
            gt_kpts = torch.rand(50, 51)  # 50 ground truth keypoints
            correct_preds = _generate_per_image_tp(predn, gt_bboxes, gt_cls, pred_kpts, gt_kpts)
            ```

        Note:
            `0.53` scale factor used in area computation is referenced from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
        """
        if pred_kpts is not None and gt_kpts is not None:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(gt_bboxes)[:, 2:].prod(1) * 0.53
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=self.sigma, area=area)
        else:  # boxes
            iou = box_iou(gt_bboxes, predn[:, :4])

        return self.create_tp_iouv_matrix(predn[:, 5], gt_cls, iou)

    def plot_val_samples(self, batch, ni):
        """Plots and saves validation set samples with predicted bounding boxes and keypoints."""
        plot_images(
            batch["img"],
            batch["img_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            kpts=batch["keypoints"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predictions for YOLO model."""
        pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds], 0)
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            kpts=pred_kpts,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, pred_kpts, save_conf, shape, file):
        """Save YOLO predn to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
            keypoints=pred_kpts,
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """Converts YOLO predictions to COCO JSON format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "keypoints": p[6:],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates object detection model using COCO JSON format."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
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
                for i, eval in enumerate([COCOeval(anno, pred, "bbox"), COCOeval(anno, pred, "keypoints")]):
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
