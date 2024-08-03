# 可以一目十行的ultralytics献给宝子们

## ALSS论文原始代码对应版本 当然我还是建议使用本仓库最新版本哈哈
论文模型配置文件 runs/ablation_experiment/ass-0.879/ASS.yaml
从来不选择最高的方案，选择演化成本最低的方案


## 运行
一键运行指令 bash train.py

## 文件解读
plot_curve 是论文中的曲线绘制原始图
plot_compraison 是对比实验图
plot_material 是绘图的原始数据
runs 包含消融实验和对比实验原始数据
runs.tar.gz 是原始数据打包文件
script 提供绘图脚本


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


### 验证模型
`self.metrics, self.fitness = self.validate()`