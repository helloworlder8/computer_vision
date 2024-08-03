# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import copy

import numpy as np
import torch.nn as nn

from ultralytics.data import creat_dataloader, create_dataset
from ultralytics.engine.engine_trainer import Engine_Trainer
from ultralytics.projects import yolo
from ultralytics.nn.tasks_model import Detection_Model,creat_model_dict_add
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
import json
import yaml
from ultralytics.utils import IterableSimpleNamespace
from pathlib import Path
class Detection_Trainer(Engine_Trainer): #目标检测的训练 好多数据都是爸爸那里拿来的 你啥也不要
    """
    A class extending the Engine_Trainer class for training based on a detection model.

   """



    def get_dataloader(self, dataset_sp, batch_size=16, rank=0, mode="train"): #得到数据加载器
        """Construct and return dataloader."""
        assert mode in ["train", "val"]
        with torch_distributed_zero_first(rank):  # init yolo_dataset *.cache only once if DDP
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32) #32
            # with open('trainer_args.json', 'w') as file:
            #     json.dump(vars(self.args), file)  # Convert Namespace to dict and save
            # data_dic_serializable = {key: str(value) if isinstance(value, Path) else value for key, value in self.data_dict.items()}
            # with open('data_dict.json', 'w') as file:
            #     json.dump(data_dic_serializable, file)
            # 参数 数据集路径 数据字典 总批次 步长 模式 矩形数据
            yolo_dataset = create_dataset(self.args, dataset_sp, self.data_dict, batch_size, stride=gs, mode=mode, rect=mode == "val")

        shuffle = mode == "train"
        if getattr(yolo_dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return creat_dataloader(yolo_dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def normalized_batch_images(self, batch_labels_list):
        """Preprocesses a batch of images by scaling and converting to float."""
        if self.data_dict["ch"]==1: #兼容单通到数据
           batch_labels_list['img'] = batch_labels_list['img'][:,:1,:,:]  
        batch_labels_list["img"] = batch_labels_list["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch_labels_list["img"]
            sz = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch_labels_list["img"] = imgs
        return batch_labels_list

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data_dict["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data_dict["nc"]  # attach number of classes to model
        self.model.names = self.data_dict["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model


    def build_model(self, model_str=None, model=None, verbose=True):
        """Return a YOLO detection task."""
        model_dict = creat_model_dict_add(model_str)
        detection_model = Detection_Model(model_dict, ch=self.data_dict["ch"], nc=self.data_dict["nc"], verbose=verbose and RANK == -1) #mark 模型 通道 数据 mdoel_dict 传参 data_dict
        if model:
            detection_model.load(model)
        return detection_model

    def get_validator(self):
        """Returns a Detection_Validator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.Detection_Validator(self.test_dataloader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["img_sp"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_result_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_dataloader.dataset.labels_dict], 0) #(2320, 4) 所有的边界框
        cls = np.concatenate([lb["cls"] for lb in self.train_dataloader.dataset.labels_dict], 0)      #(2320, 1) 所有的类别
        plot_labels(boxes, cls.squeeze(), names=self.data_dict["names"], save_dir=self.save_dir, on_plot=self.on_plot)
        # 标签 类别 保存路径 画画


def get_dataloader(dataset_sp, batch_size=16, rank=-1, mode="train"): #得到数据加载器
    """Construct and return dataloader."""
    assert mode in ["train", "val"]
    with torch_distributed_zero_first(rank):  # init yolo_dataset *.cache only once if DDP
        # gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32) #32
        with open('debug_param/trainer_args.json', 'r') as file:
            trainer_args = IterableSimpleNamespace(** json.load(file)) 
        with open('debug_param/data_dict.json', 'r') as file:
            data_dict = json.load(file)

        yolo_dataset = create_dataset(trainer_args, dataset_sp, data_dict, batch_size, stride=32, mode=mode, rect=mode == "val")

    shuffle = mode == "train"
    if getattr(yolo_dataset, "rect", False) and shuffle:
        LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    # workers = self.args.workers if mode == "train" else self.args.workers * 2
    return creat_dataloader(yolo_dataset, batch_size, 8, shuffle, rank)  # return dataloader

if __name__ == "__main__":
    get_dataloader('/home/gcsx/ANG/ultralytics-main/datasets/coco128/images/train2017')