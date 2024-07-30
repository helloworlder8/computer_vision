# 可以一目十行的ultralytics
基于ultralytics-8.2.60版本修改部分函数名和逻辑，整体架构没有改变。

## 以通过测试的部分
> yaml和pt训练模型 目标检测和语义分割 `参考test_self/main.py` 
> 目标检测提取框sam模型分割  `参考test_self/main2.py` 



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