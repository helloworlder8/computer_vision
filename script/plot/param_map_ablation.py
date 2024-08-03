import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import matplotlib.ticker as ticker
import os

# 检查文件夹是否存在，如果不存在则创建它
if not os.path.exists("plot_curve"):
    os.makedirs("plot_curve")
# 模型数据
models = ["YOLOv8-N'", 'M1', 'M2', 'M3', 'M4', 'M5(ALSS-YOLO)']
mAP = [0.874, 0.877, 0.887, 0.886, 0.889, 0.891]
parameters = [1.795, 1.482, 1.483, 1.433, 1.452, 1.452]
original_colors = [
    (254, 194, 17),
    (81,  81, 81),
    (26, 111, 223),
    (55, 173, 107),
    (117, 119, 222),
    (239, 0, 0)
]
colors = [(r/255, g/255, b/255) for r, g, b in original_colors]
shapes = ['o', 'D', '>', '*', 's', 'p']
# https://blog.csdn.net/weixin_42638388/article/details/104439779


# 创建图形和子图对象
fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')

# 在每个子图上画散点图
for i in range(len(models)):
    if parameters[i] < 1.5:
        ax.scatter(parameters[i], mAP[i], label=models[i], color=colors[i], marker=shapes[i], s=100)
        ax.text(parameters[i]+0.002, mAP[i], models[i], fontsize=9, ha='left', va='center')
    else:
        ax2.scatter(parameters[i], mAP[i], label=models[i], color=colors[i], marker=shapes[i], s=100)
        ax2.text(parameters[i]+0.002, mAP[i], models[i], fontsize=9, ha='left', va='center')
ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.025))  # 每0.02一个刻度

# 设置每个子图的x轴范围
ax.set_xlim(1.4, 1.52)
ax2.set_xlim(1.76, 1.835)

ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
# ax.yaxis.tick_left()
# ax.tick_params(labelright='off')
# 小短线
ax2.yaxis.tick_left()

plt.setp(ax.get_yticklines(), visible=False)
# 添加断轴标记
d = .015  # 断轴标记的大小
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((-d, +d), (-d, +d), **kwargs)

# 设置共享的y轴标题
ax.set_ylabel('mAP0.50',fontsize=10)

# 设置一个共享的x轴标签
fig.text(0.5, 0.02, 'Parameters(m)', ha='center',fontsize=10)  # 0.5, 0.02 是文本位置的相对坐标
ax.grid(True)
ax2.grid(True)
plt.savefig("plot_curve/ablation_mAP_PARAM.png", dpi=520)
# 显示图形
plt.show()
