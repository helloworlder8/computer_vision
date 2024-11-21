# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

import gc
import math
import os
import subprocess
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg_yaml import get_args, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import attribute_assignment, attempt_load_weights
from ultralytics.utils import (
    DEFAULT_CFG,
    LOGGER,
    RANK,
    TQDM,
    __version__,
    callbacks,
    clean_url,
    colorstr,
    emojis,
    yaml_save,
)
from ultralytics.utils.autobatch import auto_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
)
import re
import shutil
import glob
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLO

class BaseTrainer:

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):

        self.args = get_args(cfg, overrides) #大参数
        self.device = select_device(self.args.device, self.args.batch)
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0
        with torch_distributed_zero_first(RANK):
            self._check_download_dataset()
            self.trainset = self.data_dict["train"]
            self.testset = self.data_dict.get("val") or self.data_dict.get("test")
        self._check_resume(overrides) #overrides用来设置属性
        self._init_dirs()
        self._init_training_params()
        self._init_epoch_metrics()
        self._init_callbacks(_callbacks)
        if RANK == -1:
            print_args(vars(self.args))


    def _init_dirs(self): #权重路径和训练参数
        """Initialize directories."""
        self.save_dir = get_save_dir(self.args) #项目 名   任务 模式
        self.args.name = self.save_dir.name #增量名

        self.args.save_dir = str(self.save_dir)

        self.wdir = self.save_dir / "weights"
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))
        if self.args.model_name.endswith('.yaml'):
            # 使用正则表达式替换，去掉数字和 'n', 's', 'l', 'm', 'x' 后缀
            unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(self.args.model_name))  # 例如 yolov8x.yaml -> yolov8.yaml
            
            # 查找匹配的文件
            model_yaml = glob.glob(str(ROOT/ "**" /unified_path), recursive=True) or glob.glob(str(ROOT.parent / unified_path))
            data_yaml = glob.glob(str(ROOT/ "**" /str(self.args.data)), recursive=True)
            if model_yaml:  # 确保找到模型文件
                shutil.copy(model_yaml[0], self.save_dir)  # 复制模型文件到指定目录
                shutil.copy(data_yaml[0], self.save_dir)  # 复制数据文件到指定目录
            else:
                print("拷贝不了")
            # shutil.copy(self.args.data, self.save_dir)
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"




    def _init_training_params(self):
        """Initialize training parameters."""
        self.model_name = self.args.model_name
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)
        self.batch_size = self.args.batch
        self.total_epochs = self.args.epochs
        self.save_period = self.args.save_period
        self.start_epoch = 0
        self.validator = None
        self.metrics_value = None
        self.plots = {}
        self.val_interval_counter = 0  
        self.lf = None
        self.scheduler = None
        self.ema = None
        """Initialize hub."""
        self.hub_session = None

    def _init_epoch_metrics(self):
        """Initialize current_epoch level metrics."""
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.avg_loss_items = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]


    def _init_callbacks(self, _callbacks):
        """Initialize callbacks."""
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in {-1, 0}:
            callbacks.add_integration_callbacks(self)


    def add_callback(self, event: str, callback):
        """Appends the given callback."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Overrides the existing callbacks with the given callback."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        self._determine_world_size()

        if self.world_size > 1 and "LOCAL_RANK" not in os.environ:
            self._adapt_environment_for_ddp
            self._run_ddp_training()
        else:
            self._single_card_training(self.world_size)


    def _determine_world_size(self):
        """确定运行环境的world_size"""
        if isinstance(self.args.device, str) and len(self.args.device): ## i.e. device='0' or device='0,1,2,3'
            self.world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)): # # i.e. device=[0, 1, 2, 3] 
            self.world_size = len(self.args.device)
        elif torch.cuda.is_available():# default to device 0
            self.world_size = 1
        else:
            self.world_size = 0 # i.e. device='cpu' or 'mps'

    def _adapt_environment_for_ddp(self):
        """环境适配性检查与自适应调整。"""

        # Argument checks
        if self.args.rect:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
            self.args.rect = False
        if self.args.batch == -1:
            LOGGER.warning(
                "WARNING ⚠️ 'batch=-1' for AutoBatch is incompatible with Multi-GPU training, setting "
                "default 'batch=16'"
            )
            self.args.batch = 16
            

    def _run_ddp_training(self):
        """执行DDP训练的流程。"""
        cmd, file = generate_ddp_command(self.world_size, self)
        try:
            LOGGER.info(f'{colorstr("DDP:")} debug command {" ".join(cmd)}')
            subprocess.run(cmd, check=True)
        except Exception as e:
            raise e
        finally:
            ddp_cleanup(self, str(file))
            
               
    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.total_epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.total_epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        # LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size,
        )


    def _freeze_layers(self):
  
        # Determine freeze list
        freeze_list = (
            self.args.freeze if isinstance(self.args.freeze, list) else range(self.args.self.args.freeze) if isinstance(self.args.freeze, int) else []
        )
        always_freeze_names = [".dfl"]  # Always freeze these layers
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names

        # Freeze or unfreeze layers based on requirements
        for name, param in self.model.named_parameters():
            if any(freeze_name in name for freeze_name in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{name}'")
                param.requires_grad = False
            elif not param.requires_grad and param.dtype.is_floating_point:
                LOGGER.info(
                    f"WARNING ⚠️ setting 'requires_grad=True' for previously frozen layer '{name}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                param.requires_grad = True

    def _check_AMP(self, world_size):
        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        # if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
        #     callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
        #     self.amp = torch.tensor(check_amp(self.model), device=self.device)
        #     callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        if world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=False)


    def determine_batch_size(self,world_size):
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = auto_batch_size(
                model=self.model,
                imgsz=self.args.imgsz,
                amp=self.amp,
                batch=self.batch_size,
            )
        return self.batch_size // max(world_size, 1)

    def get_dataloaders_validator(self, batch_size):
        # Dataloaders
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode="train") #数据路径 batch_size 排名 模式
        if RANK in {-1, 0}:
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.testset, batch_size=batch_size if self.args.task == "obb" else batch_size * 2, rank=-1, mode="val"
            )
            self.validator = self._get_loss_names_validator()
            metric_keys = self.validator.metrics.keys + self.prefix_loss_items(prefix="val")
            self.metrics_value = dict(zip(metric_keys, [0] * len(metric_keys))) #指标字典
            self.ema = ModelEMA(self.model)
            # if self.args.plots:
            #     self.plot_training_labels() #labels_correlogram.jpg labels.jpg


    def optimizer_scheduler(self):
        # Optimizer
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.total_epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            decay=weight_decay,
            iterations=iterations,
            lr=self.args.lr0,
            momentum=self.args.momentum,
        )
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        self._setup_scheduler()
        
        
    def _setup_model_dataloaders_optimizer(self, world_size):
        """Builds dataloaders and optimizer on correct rank process."""

        # Model
        self.run_callbacks("on_pretrain_routine_start")
        ckpt = self._setup_model() #没有模型的话也拿到了模型
        self.model = self.model.to(self.device)
        self.set_model_attributes()

        self._freeze_layers()

        self._check_AMP(world_size)


        # Check imgsz9
        self.stride = stride = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=stride, floor=stride, max_dim=1)
 

        # Batch size
        batch_size =  self.determine_batch_size(world_size)

        
        self.get_dataloaders_validator(batch_size)

        self.optimizer_scheduler()

        
        self.stopper, self.stop_flag = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1 
        self.run_callbacks("on_pretrain_routine_end")



    def _single_card_training(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
         
        
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_model_dataloaders_optimizer(world_size)
        self._init_train_state(world_size)

        current_epoch = self.start_epoch 
        while True:
            self._train_epoch_prepare(current_epoch)
            self._train_epoch(world_size, current_epoch)
            self._end_epoch(current_epoch)
            if self.stop_flag:
                break
            
            current_epoch += 1
        
        self._final_eval_plot(current_epoch) #评估和画图
        self.run_callbacks("teardown")

    def _init_train_state(self, world_size):
        """Initialize the state variables and log initial info."""
        self.epoch_batch = len(self.train_loader) #一轮中多少次迭代
        self.nw = max(round(self.args.warmup_epochs * self.epoch_batch), 100) if self.args.warmup_epochs > 0 else -1 #热身多少次迭代
        self.last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.total_epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.total_epochs - self.args.close_mosaic) * self.epoch_batch
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

    def _train_epoch_prepare(self, current_epoch):
        """Prepare for a new training current_epoch."""
        self.current_epoch = current_epoch #轮次记录
        self.run_callbacks("on_train_epoch_start")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.scheduler.step()
        self.model.train() #训练模式
        if RANK != -1:
            self.train_loader.sampler.set_epoch(current_epoch)
        if current_epoch == (self.total_epochs - self.args.close_mosaic): #关闭马赛克
            self._close_dataloader_mosaic()
            self.train_loader.reset()
        if RANK in {-1, 0}:
            LOGGER.info(self.progress_string()) #打印 正式加载一批数据
            self.pbar = TQDM(enumerate(self.train_loader), total=self.epoch_batch)
        else:
            self.pbar = enumerate(self.train_loader)
        self.avg_loss_items = None

    def _train_epoch(self, world_size, current_epoch):
        """Train the model for one current_epoch."""
        for i, batch in self.pbar:
            self.run_callbacks("on_train_batch_start")
            sum_batch = i + self.epoch_batch * current_epoch
            if sum_batch <= self.nw:
                self._apply_warmup(sum_batch)
            
            with torch.cuda.amp.autocast(self.amp):
                batch = self._train_data_preprocess(batch) #归一化
                self.loss, self.loss_items = self.model(batch) # 极重点
                if RANK != -1:
                    self.loss *= world_size
                self.avg_loss_items = (self.avg_loss_items * i + self.loss_items) / (i + 1) if self.avg_loss_items is not None else self.loss_items #平均单独损失
            
            self.scaler.scale(self.loss).backward()
            
            if sum_batch - self.last_opt_step >= self.accumulate:
                self.optimizer_step()
                self.last_opt_step = sum_batch
            
            self._log_plot( batch, sum_batch)
            self.run_callbacks("on_train_batch_end")
            
            
            
            if self._timed_stopping():
                break
        
        




    def _apply_warmup(self, sum_batch):
        """Apply learning rate and momentum warmup."""
        xi = [0, self.nw]
        self.accumulate = max(1, int(np.interp(sum_batch, xi, [1, self.args.nbs / self.batch_size]).round()))
        for j, x in enumerate(self.optimizer.param_groups):
            x["lr"] = np.interp(
                sum_batch, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(self.current_epoch)]
            )
            if "momentum" in x:
                x["momentum"] = np.interp(sum_batch, xi, [self.args.warmup_momentum, self.args.momentum])

    def _log_plot(self, batch, sum_batch):
        """Log the progress of the current batch."""
        mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
        loss_len = self.avg_loss_items.shape[0] if len(self.avg_loss_items.shape) else 1
        avg_loss_items = self.avg_loss_items if loss_len > 1 else torch.unsqueeze(self.avg_loss_items, 0)
        if RANK in {-1, 0}:
            self.pbar.set_description( #打印
                ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                % (f"{self.current_epoch + 1}/{self.total_epochs}", mem, *avg_loss_items, batch["cls"].shape[0], batch["img"].shape[-1])
            )
            if self.args.plots and sum_batch in self.plot_idx:
                self.plot_training_samples(batch, sum_batch)

    def _timed_stopping(self):
        """Check if training time has exceeded the specified limit."""
        if self.args.time:
            self.stop_flag = (time.time() - self.train_time_start) > (self.args.time * 3600)
            if RANK != -1:
                broadcast_list = [self.stop_flag if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop_flag = broadcast_list[0]
        return self.stop_flag

    # def _end_epoch(self, current_epoch): #验证结果保存权重
    #     """Perform end-of-current_epoch operations."""
    #     self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)} #各组的学习率
    #     self.run_callbacks("on_train_epoch_end")
        
    #     if RANK in {-1, 0}:
    #         final_epoch = current_epoch + 1 >= self.total_epochs #false
    #         self.ema.update_attr(self.model, include=["model_dict", "nc", "args", "names", "stride", "class_weights"])
            
    #         self.val_interval_counter += 1
    #         if self.args.val and self.args.val_interval == self.val_interval_counter:
    #             self.val_interval_counter=0
    #             bool_val = True
    #         else:
    #             bool_val = False
                  
    #         if bool_val or final_epoch or self.stopper.possible_stop or self.stop_flag:
    #             self.metrics_value, self.fitness = self.validate()
    #         self.save_metrics(metrics={**self.prefix_loss_items(self.avg_loss_items), **self.metrics_value, **self.lr})
    #         self.stop_flag |= self.stopper(current_epoch + 1, self.fitness) or final_epoch
    #         if self.args.time:
    #             self.stop_flag |= (time.time() - self.train_time_start) > (self.args.time * 3600)
    #         if self.args.save or final_epoch:
    #             self.save_model()
    #             self.run_callbacks("on_model_save")
        
    #     t = time.time()
    #     self.epoch_time = t - self.epoch_time_start
    #     self.epoch_time_start = t
    #     if self.args.time:
    #         mean_epoch_time = (t - self.train_time_start) / (current_epoch - self.start_epoch + 1)
    #         self.total_epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
    #         self._setup_scheduler()
    #         self.scheduler.last_epoch = self.current_epoch
    #         self.stop_flag |= current_epoch >= self.total_epochs
    #     self.run_callbacks("on_fit_epoch_end")
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     if RANK != -1:
    #         broadcast_list = [self.stop_flag if RANK == 0 else None]
    #         dist.broadcast_object_list(broadcast_list, 0)
    #         self.stop_flag = broadcast_list[0]




    def validate_save_metrics_model(self, current_epoch):
        """Handles validation checks and stopping conditions."""

        self.ema.update_attr(self.model, include=["model_dict", "nc", "args", "names", "stride", "class_weights"])
        
        final_epoch = current_epoch + 1 >= self.total_epochs 
        self.val_interval_counter += 1
        bool_val = self.args.val and self.args.val_interval == self.val_interval_counter
        if final_epoch or bool_val or  self.stopper.possible_stop or self.stop_flag:
            self.metrics_value, self.fitness = self.validate()
            
        self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
        self.save_metrics(metrics={**self.prefix_loss_items(self.avg_loss_items), **self.metrics_value, **self.lr})
        
        self.stop_flag |= self.stopper(current_epoch + 1, self.fitness) or final_epoch
        if self.args.time:
            self.stop_flag |= (time.time() - self.train_time_start) > (self.args.time * 3600)

        # Save model
        if self.args.save or final_epoch:
            self.save_model()
            self.run_callbacks("on_model_save")
            

    def handle_time_epoch_update(self, current_epoch):
        """Handles time tracking and epoch adjustments."""
        t = time.time()
        self.epoch_time = t - self.epoch_time_start
        self.epoch_time_start = t
        if self.args.time:
            mean_epoch_time = (t - self.train_time_start) / (current_epoch - self.start_epoch + 1)
            self.total_epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
            self._setup_scheduler()
            self.scheduler.last_epoch = self.current_epoch
            self.stop_flag |= current_epoch >= self.total_epochs

    def _end_epoch(self, current_epoch):
        """Main function to handle end-of-epoch operations."""
        self.run_callbacks("on_train_epoch_end")
        
        if RANK in {-1, 0}:  # Proceed with validation and stopping logic if master node
            self.validate_save_metrics_model(current_epoch)
        self.handle_time_epoch_update(current_epoch)
        self.run_callbacks("on_fit_epoch_end")
        
        # Cleanup and memory management
        gc.collect()
        torch.cuda.empty_cache()

        if RANK != -1:  # Synchronize stop flag across all ranks
            broadcast_list = [self.stop_flag if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            self.stop_flag = broadcast_list[0]






    def _final_eval_plot(self, current_epoch):
        """Finalize training and perform any necessary cleanup."""
        if RANK in {-1, 0}:
            LOGGER.info(
                f"\n{current_epoch - self.start_epoch + 1} epochs completed in "
                f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
            )
            self.final_eval()
            if self.args.plots:
                self.plot_results()
            self.run_callbacks("on_train_end")
        gc.collect()
        torch.cuda.empty_cache()


    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        import pandas as pd  # scope for faster 'import ultralytics'

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.current_epoch, #当前轮次
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics_value, **{"fitness": self.fitness}},
                "train_results": {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()},
                "date": datetime.now().isoformat(),
                "version": __version__,
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.current_epoch > 0) and (self.current_epoch % self.save_period == 0):
            (self.wdir / f"current_epoch{self.current_epoch}.pt").write_bytes(serialized_ckpt)  # save current_epoch, i.e. 'epoch3.pt'

    # def _create_data_dict(self):

    #     try:
    #         if self.args.task == "classify":
    #             data_dict = check_cls_dataset(self.args.data)
    #         elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
    #             "detect",
    #             "segment",
    #             "pose",
    #             "obb",
    #         }:
    #             data_dict = check_det_dataset(self.args.data)
    #             if "data_yaml" in data_dict:
    #                 self.args.data = data_dict["data_yaml"]  # 给全称
    #     except Exception as e:
    #         raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
    #     self.data_dict = data_dict
    #     return data_dict["train"], data_dict.get("val") or data_dict.get("test")

    def _check_download_dataset(self):
        try:
            if self.args.task == "classify":
                self.data_dict = check_cls_dataset(self.args.data)
            elif self.args.data.split(".")[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                self.data_dict = check_det_dataset(self.args.data)
                if "data_yaml" in self.data_dict:
                    self.args.data = self.data_dict["data_yaml"]  # 给全称
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        
    def _setup_model(self):
        """Load/create/download model for any task."""
        if hasattr(self, 'model') and isinstance(self.model, torch.nn.Module):  # if self has model attribute and it's an instance of torch.nn.Module
            return #直接有模型情况
        weights = None
        cfg = self.model_name #这个是yaml文件情况
        ckpt = None
        if str(self.model_name).endswith(".pt"):
            ckpt, weights = attribute_assignment(self.model_name)
            if hasattr(weights, 'yaml'):
                cfg = weights.model_dict = weights.yaml
            else:
                cfg = weights.model_dict #有预训练权重的情况
        elif isinstance(self.args.pretrained, (str, Path)):
            _, weights = attribute_assignment(self.args.pretrained)
            
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def _train_data_preprocess(self, batch):
        """Allows custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator.

        The returned dict is expected to contain "fitness" key.
        """
        metrics_value = self.validator(self)
        fitness = metrics_value.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics_value, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def _get_loss_names_validator(self):
        """Returns a NotImplementedError when the _get_loss_names_validator function is called."""
        raise NotImplementedError("_get_loss_names_validator function not implemented in trainer")

    def get_dataloader(self, dataset_str, batch_size=16, rank=0, mode="train"):
        """Returns dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def _build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("_build_dataset function not implemented in trainer")

    def prefix_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """To set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = "" if self.csv.exists() else (("%23s," * n % tuple(["current_epoch"] + keys)).rstrip(",") + "\n")  # header
        with open(self.csv, "a") as f:
            f.write(s + ("%23.5g," * n % tuple([self.current_epoch + 1] + vals)).rstrip(",") + "\n")

    def plot_results(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.plots = self.args.plots
                    self.metrics_value = self.validator(model=f)
                    self.metrics_value.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")

    def _check_resume(self, overrides): #可以改动的
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume_pt = self.args.resume_pt
        resume = self.args.resume #默认false
        if resume_pt:
            try:
                exists = isinstance(resume_pt, (str, Path)) and Path(resume_pt).exists()
                resume_pt = Path(check_file(resume_pt) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                train_args = attempt_load_weights(resume_pt).args
                if not Path(train_args["data"]).exists():
                    train_args["data"] = self.args.data #相当于可以另外传入参数

                resume = True #返回成标记
                self.args = get_args(train_args) #
                self.args.model_name = self.args.resume_pt = str(resume_pt)  # reinstate model
                for k in "imgsz", "epochs", "batch", "device":  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None:
            return
        # 有权重直接叠加训练
        # Load start epoch and best fitness
        start_epoch = ckpt["epoch"] + 1
        best_fitness = ckpt.get("best_fitness", 0.0)

        # Resume optimizer state
        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        # Resume EMA state if applicable
        if self.ema and "ema" in ckpt and ckpt.get("ema"):
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]

        # Assert that there is something to resume if explicitly requested
        if self.resume:
            assert start_epoch < self.total_epochs, (
                f"{self.args.model_name} training to {self.total_epochs} epochs is finished, nothing to resume.\n"
                f"Consider starting a new training without resuming, i.e., 'yolo train model={self.args.model_name}'."
            )
            LOGGER.info(f"Resuming training from epoch {start_epoch} to {self.total_epochs} total epochs.")

        # Adjust total epochs for fine-tuning if starting from a ckpt
        if start_epoch > self.total_epochs:
            self.total_epochs += ckpt["epoch"]
            LOGGER.info(
                f"{self.model} has already been trained for {ckpt['epoch']} epochs. "
                f"Fine-tuning for an additional {self.total_epochs} epochs."
            )


        # Update class attributes
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch

        # Close dataloader mosaic if applicable
        if start_epoch > (self.total_epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()




    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=self.args)

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):


        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f"[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto]."
                "To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)'
        )
        return optimizer
