# 可以一目十行的ultralytics献给宝子们
基于ultralytics-8.2.60版本修改部分函数名和逻辑，整体架构没有改变。
从来不选择最高的方案，选择演化成本最低的方案

## 此版本更新说明
> 删去了AMP验证部分
> 添加了ALSS-YOLO网络结构 改写成语义分割模型通过测试
> 修改部分bug具体啥忘了

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






## 训练结果解析
[detect](results/datect.md)