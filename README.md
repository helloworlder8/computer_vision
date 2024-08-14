# 可以一目十行的ultralytics献给宝子们
基于ultralytics-8.2.60版本修改部分函数名和逻辑，整体架构没有改变。
从来不选择最高的方案，选择演化成本最低的方案

## 此版本更新说明
> 删去了AMP验证部分

> 添加了ALSS-YOLO网络结构

> 训练验证测试全通过，支持原生单通道训练，支持通过配置文件传参修改IOU算法`

> #### 支持最新版本SAM2模型推理

## 以通过测试的部分
> `script/bug_test.py`



## 重要函数汇总


### 训练入口
`self.trainer.train()`


### 单卡训练

`def _single_card_training(self, world_size=1):`


### 获取数据集
`self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode="train")`


### 数据通过模型
`self.loss, self.loss_items = self.model(batch)`


### 分割检测头
`class Segment(Detect):`


### 语义分割损失计算
`class SegmentationLoss(DetectionLoss):`

### 解析模型
`def parse_model(model_dict, ch, verbose=True):`


### 验证模型
`self.metrics, self.fitness = self.validate()`


## IOU改进
支持通过传参改变iou
`def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False,FineSIoU= False, WIoU=False, Focal=False, pow=1, gamma=0.5, scale=False, eps=1e-7):`


## 画图
```python
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)
```