# 可以一目十行的ultralytics献给宝子们
基于ultralytics-8.2.60版本

此版本提供预训练权重文件将在之后版本删除
修改部分函数名和逻辑，整体架构没有改变。
从来不选择最高的方案，选择演化成本最低的方案
修改代码尽量保证和官网提供的外部接口一致



## 验证参数解读

#### 混淆矩阵中的置信度默认0.25  iou默认为0.45
```python
self.confusion_matrix_conf = 0.25 if conf in {None, 0.001} else conf  # apply 0.25 if default val conf is passed
self.confusion_matrix_iou = confusion_matrix_iou #iou默认0.45
```


#### 验证集中预测框非极大值抑制的置信度为0.001  非极大值抑制iou为0.7  注意这个iou和常规理解的iou不同，仅仅作为一个阈值 此步骤的非极大值抑制是为了过滤部分预测信息 包扩bbox cls置信度 cls索引 mask信息（语义分割）

```python
assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
```

#### 代码给出的精度召回率是在    iou为0.5的情况下最佳F1值对应的置信度下的值
```python
# Recall
recall = tpc / (n_l + eps)  # recall curve #(3737, 10)
r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases   iou阈值是0.5

# Precision
precision = tpc / (tpc + fpc)  # precision curve
p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)  # p at pr_score

i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]  # max-F1 precision, recall, F1 values
tp = (r * unique_gt_cls_num).round()  # true positives
fp = (tp / (p + eps) - tp).round()  # false positives
```


## 此版本更新说明
> 删去了AMP验证部分

> 添加了ALSS-YOLO网络结构

> 训练验证测试全通过，支持原生单通道训练，支持通过配置文件传参修改IOU算法`

> #### 支持最新版本SAM2模型推理
> #### 支持最新版本FastSAM模型推理


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


### 前向传播
`x = m(x)  # run`

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