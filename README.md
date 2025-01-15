
<h1 align="center">
    <p>可以一目十行的ultralytics献给宝子们</p>
</h1>

📝训练验证可视化以及数据转换等自动化脚本目录 scripts

🤗 **论文引用以及相关介绍**
* [ALSS-YOLO: An Adaptive Lightweight Channel Split and Shuffling Network for TIR Wildlife Detection in UAV Imagery](docs/ALSS-YOLO.md)
📝[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10680397)

* [Iterative Optimization Annotation Pipeline and ALSS-YOLO-Seg for Efficient Banana Plantation Segmentation in UAV Imagery](docs/ALSS-YOLO-seg.md)
📝[paper](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2024.1508549/abstract)

## 文档快速索引
[评估指标](docs/Metrics.md)

[显示](docs/Display.md)

[数据集加载和处理](docs/Datasets.md)

## 已通过测试的部分
> `script/bug_test.py`



## 重要函数汇总


### 训练入口
- self.trainer.train()


### 单卡训练
- def _single_card_training(self, world_size=1):


### 获取数据集
- self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode="train")


### 数据通过模型
- self.loss, self.loss_items = self.model(batch)
- def loss(self, batch, preds=None): 计算损失
- x = m(x)  # 前向传播

### 分割检测头
- class Segment(Detect):


### 语义分割损失计算
- class SegmentationLoss(DetectionLoss):

### 解析模型
- def parse_model(model_dict, ch, verbose=True):


### 验证模型
- self.metrics, self.fitness = self.validate()


### 前向传播
- x = m(x)  # run

### IOU改进
支持通过传参改变iou
- def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False,FineSIoU= False, WIoU=False, Focal=False, pow=1, gamma=0.5, scale=False, eps=1e-7):


### 推理阶段
- preds = self.inference(new_img, *args, **kwargs)



### 分割头
- elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}



### 预测置信度
- custom = {"conf": 0.4, "batch": 1, "save": False, "mode": "predict"}



