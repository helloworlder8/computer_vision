import matplotlib.pyplot as plt
import pandas as pd
import os

# 检查文件夹是否存在，如果不存在则创建它
if not os.path.exists("plot_curve"):
    os.makedirs("plot_curve")
# 绘制PR
def plot_PR():
    pr_csv_dict = {
        "YOLOv9t-Seg'": r'result/Comparative_experiment_exp_val/yolov9t/MaskPR_curve.csv',
        "YOLOv8-p6-Seg'": r'result/Comparative_experiment_exp_val/yolov8-seg-p6/MaskPR_curve.csv',
        "YOLOv6-Seg'": r'result/Comparative_experiment_exp_val/yolov6-seg/MaskPR_curve.csv',
        "YOLOv8-ghost-Seg'": r'result/Comparative_experiment_exp_val/yolov8-seg-ghost/MaskPR_curve.csv',
        'ALSS-YOLO-Seg': r'result/Comparative_experiment_exp_val/ALSSn/MaskPR_curve.csv',
        # 'ALSS-YOLO-Seg': r'result/Ablation_experiment_exp_val/M4/MaskPR_curve.csv',
        # 'M6': r'result/Ablation_experiment_exp_val/M1/MaskPR_curve.csv',
    }

    # 指定颜色列表（红橙黄绿蓝靛紫）
    # colors = [
    #     (223/255, 141/255, 143/255),
    #     (255/255, 170/255, 137/255),
    #     (126/255, 208/255, 248/255),
    #     (252/255, 211/255, 181/255),
    #     (126/255, 126/255, 237/255),
    #     (255/255, 220/255, 126/255)
    # ]
    original_colors = [
        (254, 194, 17),
        (81,  81, 81),
        (26, 111, 223),
        # (55, 173, 107),
        (117, 119, 222),
        (239, 0, 0),
        # (102, 76, 98)
    ]
    colors = [(r/255, g/255, b/255) for r, g, b in original_colors]
    # 绘制 PR 曲线
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
    for modelname, color in zip(pr_csv_dict, colors):
        res_path = pr_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[3]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth=2, color=color)

    # 添加x轴和y轴标签
    ax.set_xlabel('Recall',fontsize=12)
    ax.set_ylabel('Precision',fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.legend()
    # plt.grid(True)  # 显示网格线
    # 显示图像
    fig.savefig("plot_curve/Ablation_PR.png", dpi=400)
#     plt.show()



# 绘制F1
def plot_F1():
    f1_csv_dict = {
        "YOLOv9t-Seg'": r'result/Comparative_experiment_exp_val/yolov9t/MaskF1_curve.csv',
        "YOLOv8-p6-Seg'": r'result/Comparative_experiment_exp_val/yolov8-seg-p6/MaskF1_curve.csv',
        "YOLOv6-Seg'": r'result/Comparative_experiment_exp_val/yolov6-seg/MaskF1_curve.csv',
        "YOLOv8-ghost-Seg'": r'result/Comparative_experiment_exp_val/yolov8-seg-ghost/MaskF1_curve.csv',
        "ALSS-YOLO-Seg": r'result/Comparative_experiment_exp_val/ALSSn/MaskF1_curve.csv',


    }
    original_colors = [
        (254, 194, 17),
        (81,  81, 81),
        (26, 111, 223),
        # (55, 173, 107),
        (117, 119, 222),
        (239, 0, 0),
        # (102, 76, 98)
    ]
    colors = [(r/255, g/255, b/255) for r, g, b in original_colors]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)


    for modelname, color in zip(f1_csv_dict, colors):
        res_path = f1_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[3]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth=2, color=color)


    # 添加x轴和y轴标签
    ax.set_xlabel('Confidence',fontsize=12)
    ax.set_ylabel('F1',fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend()
    # plt.grid()  # 显示网格线
    # 显示图像
    fig.savefig("plot_curve/Ablation_F1.png", dpi=400)
    # plt.show()

def plot_MAP():
    map_csv_dict = {
        "YOLOv5-N'": r'plot_material/comparsion_experiment/yolov5-0.876_results.csv',
        'YOLOv8-ghost': r'plot_material/comparsion_experiment/yolov8-ghost-0.879_results.csv',
        'YOLOv8-AM': r'plot_material/comparsion_experiment/yolov8_ECA3-0.889_results.csv',
        'YOLOv8-p2': r'plot_material/comparsion_experiment/yolov8-p2-0.900_results.csv',
        'ALSS-YOLO-m': r'plot_material/comparsion_experiment/ASS-YOLO-m_results.csv',
        'ALSS-YOLO': r'plot_material/ablation_experiment/ass-focus-poolconv-atta-iou-0.891_results.csv',
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
    fig.savefig("plot_curve/Ablation_mAP0.5.png", dpi=520)


if __name__ == '__main__':
    # plot_PR()   # 绘制PR
    plot_F1()   # 绘制F1
    # plot_MAP()