# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolov8n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolov8n.pt                 # PyTorch
                          yolov8n.torchscript        # TorchScript
                          yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolov8n_openvino_model     # OpenVINO
                          yolov8n.engine             # TensorRT
                          yolov8n.mlpackage          # CoreML (macOS-only)
                          yolov8n_saved_model        # TensorFlow SavedModel
                          yolov8n.pb                 # TensorFlow GraphDef
                          yolov8n.tflite             # TensorFlow Lite
                          yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolov8n_paddle_model       # PaddlePaddle
                          yolov8n_ncnn_model         # NCNN
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg_yaml import get_args, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


class BaseValidator:
    """
    BaseValidator.

    A base class for creating validators.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary.
        device (torch.device): Device to use for validation.
        batch_index (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names.
        seen: Records the number of images seen so far during validation.
        stats: Placeholder for statistics during validation.
        confusion_matrix: Placeholder for a confusion matrix.
        nc: Number of classes.
        iouv: (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (dict): Dictionary to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective
                      batch processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
    """
        # super().__init__(args, dataloader, save_dir, pbar, _callbacks)
    def __init__(self, args=None, dataloader=None, save_dir=None, pbar=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        self.args = get_args(overrides=args) #大参数
        self.dataloader = dataloader
        self.pbar = pbar
        self.stride = None
        self.data_dict = None
        self.device = None
        self.batch_index = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats_dict = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.001  # default conf=0.001
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):#model模型路径权重文件或者  真实的模型
        """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data_dict = trainer.data_dict
            self.args.half = self.device.type != "cpu"  # force FP16 val during training
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.current_epoch == trainer.total_epochs - 1)
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                models=model or self.args.model_name,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half    false
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine  #32 true
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch
            elif not pt and not jit:
                self.args.batch = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data_dict = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data_dict = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data_dict.get(self.args.split), self.args.batch) #验证的时候还没有数据集得重新弄

            model.eval()
            ch = self.data_dict.get("ch", 3)
            model.warmup(imgsz=(1 if pt else self.args.batch, ch, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_index, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_index = batch_index
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds) #torch.Size([228, 6])torch.Size([19, 6])
                # 置信非极大值抑制  默认非极大值抑制置信度为0.001  非极大值抑制iou为0.7
            self.update_metrics_per_batch(preds, batch)
            if self.args.plots and batch_index < 3:
                self.plot_val_samples(batch, batch_index)
                self.plot_predictions(batch, preds, batch_index)

            self.run_callbacks("on_val_batch_end")
        stats = self.generate_metrics_info()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_add_metrics()
        self.print_write_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.prefix_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
                % tuple(self.speed.values())
            )
            LOGGER.info(f'FPS:{(1000/self.args.batch / sum(self.speed.values())):.2f}')
            
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def create_tp_iouv_matrix(self, pd_cls, tgt_cls, bbox_iou, use_scipy=False):
        """
        Matches predictions to ground truth objects (pd_cls, tgt_cls) using IoU.

        Args:
            pd_cls (torch.Tensor): Predicted class indices of shape(N,).
            tgt_cls (torch.Tensor): Target class indices of shape(M,).
            bbox_iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): tp_iouv_matrix tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - predn, 10 - IoU thresholds
        tp_iouv_matrix = np.zeros((pd_cls.shape[0], self.iouv.shape[0])).astype(bool) #(300, 10)
        # LxD matrix where L - labels (rows), D - predn (columns)
        TF = tgt_cls[:, None] == pd_cls #torch.Size([8, 300])
        bbox_iou = bbox_iou * TF  # 类别过滤
        bbox_iou = bbox_iou.cpu().numpy() #torch.Size([8, 300])
        for i, thres_i in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = bbox_iou * (bbox_iou >= thres_i)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        tp_iouv_matrix[detections_idx[valid], i] = True
            else:
                iou_posi = np.nonzero(bbox_iou >= thres_i)  # IoU > thres_i and classes match 
                iou_posi = np.array(iou_posi).T #->(24, 2)
                if iou_posi.shape[0]:
                    if iou_posi.shape[0] > 1:
                        iou_posi = iou_posi[bbox_iou[iou_posi[:, 0], iou_posi[:, 1]].argsort()[::-1]] #(值排序 索引排序 -》(20, 2)
                        iou_posi = iou_posi[np.unique(iou_posi[:, 1], return_index=True)[1]] #不要存在一个真实框匹配多个预测框的情况
                        # iou_posi = iou_posi[iou_posi[:, 2].argsort()[::-1]]
                        iou_posi = iou_posi[np.unique(iou_posi[:, 0], return_index=True)[1]] #也不要存在多个预测框匹配一个真实框的情况
                    tp_iouv_matrix[iou_posi[:, 1].astype(int), i] = True #预测情况
        return torch.tensor(tp_iouv_matrix, dtype=torch.bool, device=pd_cls.device)

    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_str, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def _build_dataset(self, img_path):
        """Build dataset."""
        raise NotImplementedError("_build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        return batch

    def postprocess(self, preds):
        """Describes and summarizes the purpose of 'postprocess()' but no details mentioned."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics_per_batch(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        pass

    def finalize_add_metrics(self, *args, **kwargs):
        """Finalizes and returns all metrics."""
        pass

    def generate_metrics_info(self):
        """Returns statistics about the model's performance."""
        return {}

    def check_stats(self, stats):
        """Checks statistics."""
        pass

    def print_write_results(self):
        """Prints the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Returns the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    # TODO: may need to put these following functions into callback
    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plots YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass
