## 验证参数解读


### 实例分割评估
SegmentMetrics


#### 混淆矩阵中的置信度默认0.25  iou默认为5
```python
self.confusion_matrix_conf = 0.25 if conf in {None, 0.001} else conf  # apply 0.25 if default val conf is passed
self.confusion_matrix_iou = confusion_matrix_iou #iou默认5
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