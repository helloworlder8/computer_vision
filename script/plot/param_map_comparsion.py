import matplotlib.pyplot as plt
import os

# 检查文件夹是否存在，如果不存在则创建它
if not os.path.exists("plot_curve"):
    os.makedirs("plot_curve")
# 假定的数据集
labels = ["YOLOv5-n'", 'YOLOv8-ghost', 'YOLOv8-AM', 'YOLOv8-p2', 'ALSS-YOLO-m', 'ALSS-YOLO']
mAP = [0.876, 0.877, 0.886, 0.894, 0.903, 0.891]
parameters = [1.518, 1.712, 1.583, 2.927, 2.924, 1.452]
original_colors = [
    (254, 194, 17),
    (81,  81, 81),
    (26, 111, 223),
    (55, 173, 107),
    (117, 119, 222),
    (239, 0, 0)
]
colors = [(r/255, g/255, b/255) for r, g, b in original_colors]
marker = ['o', 'D', '>', '*', 's', 'p']

# 创建图表
plt.figure(figsize=(10, 6))

# 添加数据点
for i in range(len(labels)):
    plt.scatter(parameters[i], mAP[i], label=labels[i], color=colors[i], marker=marker[i], s=100)
    if parameters[i] < 1.8:
        plt.text(parameters[i]+0.02, mAP[i], labels[i], fontsize=9, ha='left', va='center')
    else:
        plt.text(parameters[i]-0.02, mAP[i], labels[i], fontsize=9, ha='right', va='center')
# 添加图例
plt.legend()

# 添加标签和标题
plt.xlabel('Parameters (m)',fontsize=10)
plt.ylabel('mAP0.50',fontsize=10)
# plt.title('Comparison of mAP@0.50 and parameters of different lightweight detectors')

# 显示网格
plt.grid(True)

# 显示图表
plt.savefig("plot_curve/comparsion_mAP_PARAM.png", dpi=520)
