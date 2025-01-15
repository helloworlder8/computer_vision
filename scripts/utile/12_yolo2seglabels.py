from pathlib import Path
from ultralytics import SAM, YOLO
import torch


# # 定义检测模型和SAM模型的路径
# seg_model_pt="result/Comparative_experiment_exp/yolov9t-seg_0.849/weights/last.pt"
# output_labels_path = "./result/13/yolov9tlabels"

# seg_model_pt="result/Comparative_experiment_exp/yolov8-seg-p6_0.851/weights/last.pt"
# output_labels_path = "./result/13/yolov8p6labels"

# seg_model_pt="result/Comparative_experiment_exp/yolov6-seg_0.858/weights/last.pt"
# output_labels_path = "./result/13/yolov6labels"

# seg_model_pt="result/Comparative_experiment_exp/yolov8-seg-ghost_0.849/weights/last.pt"
# output_labels_path = "./result/13/yolov8ghostlabels"

seg_model_pt="result/Ablation_experiment_exp/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt"
output_labels_path = "./result/13/ALSSnlabels"



# 根据CUDA是否可用选择设备
device = '0' if torch.cuda.is_available() else 'cpu'

img_str = 'result/13'
# 获取图像数据路径
img_path = Path(img_str)


# 初始化检测模型和SAM模型
seg_model = YOLO(seg_model_pt)  





Path(output_labels_path).mkdir(exist_ok=True, parents=True)

# 对图像数据进行检测
seg_results = seg_model(img_path, stream=True, device=device)
print("yes")

for result in seg_results:
    # 获取类别ID
    class_ids = result.boxes.cls.int().tolist()  # noqa
    # 如果有检测到物体
    if len(class_ids):
        # 获取检测框坐标
        masks = result.masks.xyn  # Boxes object for bbox outputs  #torch.Size([6, 4])

        with open(f"{Path(output_labels_path) / Path(result.path).stem}.txt", "w") as f:
            for i in range(len(masks)):
                s = masks[i]
                # 如果分割区域为空，则跳过
                if len(s) == 0:
                    continue
                # 将分割区域坐标转换为字符串格式
                masks_i = map(str, masks[i].reshape(-1).tolist())
                # 写入标注信息
                f.write(f"{class_ids[i]} " + " ".join(masks_i) + "\n")