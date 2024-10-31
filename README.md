# 可以一目十行的ultralytics献给宝子们
官方文档 https://docs.ultralytics.com/zh/usage/simple-utilities/?h=yolo_bbox2segment#convert-coco-into-yolo-format

此版本保留所有训练结果



此版本对应论文《Iterative Optimization Annotation Pipeline and ALSS-YOLO-Seg for Efficient Banana Plantation Segmentation in UAV Imagery 》暂定
消融实验配置参考 result/Ablation_experiment
对比实验配置参考 result/Comparative_experiment
最终权重文件 ALSS-YOLO-Seg.pt
训练验证可视化以及数据转换等脚本 scripts

实验用的数据集下一个版本给出对应链接

## 文档快速索引
[评估指标](docs/Metrics.md)

[显示](docs/Display.md)

[数据集加载和处理](docs/Datasets.md)
## 以通过测试的部分
> `script/bug_test.py`



## 重要函数汇总（重点在训练）


### 训练入口
`self.trainer.train()`


### 单卡训练

`def _single_card_training(self, world_size=1):`


### 获取数据集
`self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode="train")`


### 数据通过模型
`self.loss, self.loss_items = self.model(batch)`
`def loss(self, batch, preds=None): `计算损失

### 分割检测头
`class Segment(Detect):`


### 语义分割损失计算
`class SegmentationLoss(DetectionLoss):`

### 解析模型
`def parse_model(model_dict, ch, verbose=True):`


### 验证模型
`self.metrics, self.fitness = self.validate()`


### 前向传播
`x = m(x)  # run`

## IOU改进
支持通过传参改变iou
`def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False,FineSIoU= False, WIoU=False, Focal=False, pow=1, gamma=0.5, scale=False, eps=1e-7):`


## 推理阶段
`preds = self.inference(new_img, *args, **kwargs)`



## 分割头
`elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}`



## 预测置信度
`custom = {"conf": 0.4, "batch": 1, "save": False, "mode": "predict"}`



