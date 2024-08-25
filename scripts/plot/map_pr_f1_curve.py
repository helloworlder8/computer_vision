import matplotlib.pyplot as plt
import pandas as pd
import os

# 检查文件夹是否存在，如果不存在则创建它
if not os.path.exists("plot_curve"):
    os.makedirs("plot_curve")

# 绘制PR
def plot_PR():
    pr_csv_dict = {
        "YOLOv8-N'": r'plot_material/ablation_experiment/yolov8-width_PR_curve.csv',
        'M1': r'plot_material/ablation_experiment/ass-0.879_PR_curve.csv',
        'M2': r'plot_material/ablation_experiment/ass-focus-0.887_PR_curve.csv',
        'M3': r'plot_material/ablation_experiment/ass-focus-poolconv-atta-iou-0.891_PR_curve.csv',
        'M4': r'plot_material/ablation_experiment/ass-focus-poolconv-atta-iou-0.885_PR_curve.csv',
        'M5(ALSS-YOLO)': r'plot_material/ablation_experiment/ass-focus-poolconv-0.886_PR_curve.csv',
    }

    original_colors = [
        (254, 194, 17),
        (81,  81, 81),
        (26, 111, 223),
        (55, 173, 107),
        (117, 119, 222),
        (239, 0, 0)
    ]
    colors = [(r/255, g/255, b/255) for r, g, b in original_colors]
    # 绘制 PR 曲线
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
    for modelname, color in zip(pr_csv_dict, colors):
        res_path = pr_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[6]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth=2, color=color)

    # 添加x轴和y轴标签
    ax.set_xlabel('Recall',fontsize=12)
    ax.set_ylabel('Precision',fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.legend()
    plt.grid(True)  # 显示网格线
    # 显示图像
    fig.savefig("plot_curve/ablation_PR.png", dpi=520)
#     plt.show()



# 绘制F1
def plot_F1():
    f1_csv_dict = {
        "YOLOv8-N'": r'plot_material/ablation_experiment/yolov8-width_F1_curve.csv',
        'M1': r'plot_material/ablation_experiment/ass-0.879_F1_curve.csv',
        'M2': r'plot_material/ablation_experiment/ass-focus-0.887_F1_curve.csv',
        'M3': r'plot_material/ablation_experiment/ass-focus-poolconv-atta-iou-0.891_F1_curve.csv',
        'M4': r'plot_material/ablation_experiment/ass-focus-poolconv-atta-iou-0.885_F1_curve.csv',
        'M5(ALSS-YOLO)': r'plot_material/ablation_experiment/ass-focus-poolconv-0.886_F1_curve.csv',
    }
    original_colors = [
        (254, 194, 17),
        (81,  81, 81),
        (26, 111, 223),
        (55, 173, 107),
        (117, 119, 222),
        (239, 0, 0)
    ]
    colors = [(r/255, g/255, b/255) for r, g, b in original_colors]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)


    for modelname, color in zip(f1_csv_dict, colors):
        res_path = f1_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[6]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth=2, color=color)


    # 添加x轴和y轴标签
    ax.set_xlabel('Confidence',fontsize=12)
    ax.set_ylabel('F1',fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend()
    plt.grid()  # 显示网格线
    # 显示图像
    fig.savefig("plot_curve/ablation_F1.png", dpi=520)
    # plt.show()

def plot_MAP():
    map_csv_dict = {
        "YOLOv8-N'": r'plot_material/ablation_experiment/yolov8-width_results.csv',
        'M1': r'plot_material/ablation_experiment/ass-0.879_results.csv',
        'M2': r'plot_material/ablation_experiment/ass-focus-poolconv-0.886_results.csv',
        'M3': r'plot_material/ablation_experiment/ass-focus-poolconv-atta-iou-0.891_results.csv',
        'M4': r'plot_material/ablation_experiment/ass-focus-poolconv-atta-iou-0.885_results.csv',
        'M5(ALSS-YOLO)': r'plot_material/ablation_experiment/ass-focus-poolconv-atta-iou-0.891_results.csv',
    }
    original_colors = [
        (254, 194, 17),
        (81,  81, 81),
        (26, 111, 223),
        (55, 173, 107),
        (117, 119, 222),
        (239, 0, 0)
    ]
    colors = [(r/255, g/255, b/255) for r, g, b in original_colors]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
    for modelname, color in zip(map_csv_dict, colors):
        res_path = map_csv_dict[modelname]
        data = pd.read_csv(res_path)
        plt.plot(data['       metrics/mAP50(B)'], label=modelname,color=color) #一列一列的读取 标签为键
    ax.set_xlabel('epoch',fontsize=12)
    ax.set_ylabel('mAP0.50',fontsize=12)
    # plt.xlabel('epoch')
    # plt.title('mAP_0.5')
    plt.grid()  # 显示网格线
    plt.legend()
    fig.savefig("plot_curve/ablation_mAP0.5.png", dpi=520)


if __name__ == '__main__':
    plot_PR()   # 绘制PR
    plot_F1()   # 绘制F1
    plot_MAP()