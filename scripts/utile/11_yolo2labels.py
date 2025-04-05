from pathlib import Path
from ultralytics import SAM, YOLO
import torch


# # 定义检测模型和SAM模型的路径
det_model_pt="ADE20K_detect/Javeri_detect_0.993/weights/best.pt"
output_labels_path = "../datasets/Javeri_det_seg/valid_detect_train/labels"

# det_model_pt="ADE20K_detect/Javeri_detect+10%_0.954/weights/best.pt"
# output_labels_path = "../datasets/Javeri_det_seg/valid_detect_train+10%/labels"

# det_model_pt="ADE20K_detect/Javeri_detect+20%_0.959/weights/best.pt"
# output_labels_path = "../datasets/Javeri_det_seg/valid_detect_train+20%/labels"

# det_model_pt="ADE20K_detect/Javeri_detect+50%_0.989/weights/best.pt"
# output_labels_path = "../datasets/Javeri_det_seg/valid_detect_train+50%/labels"
# 根据CUDA是否可用选择设备
device = '0' if torch.cuda.is_available() else 'cpu'

img_str = '../datasets/Javeri_det_seg/valid_detect/images/'
# 获取图像数据路径
img_path = Path(img_str)


# 初始化检测模型和SAM模型
det_model = YOLO(det_model_pt)  





Path(output_labels_path).mkdir(exist_ok=True, parents=True)

# 对图像数据进行检测
det_results = det_model(img_path, stream=True, device=device)
print("yes")

for result in det_results:
    # 获取类别ID
    class_ids = result.boxes.cls.int().tolist()  # noqa
    # 如果有检测到物体
    if len(class_ids):
        # 获取检测框坐标
        boxes = result.boxes.xywhn  # Boxes object for bbox outputs  #torch.Size([6, 4])

        with open(f"{Path(output_labels_path) / Path(result.path).stem}.txt", "w") as f:
            for i in range(len(boxes)):
                s = boxes[i]
                # 如果分割区域为空，则跳过
                if len(s) == 0:
                    continue
                # 将分割区域坐标转换为字符串格式
                boxes_i = map(str, boxes[i].reshape(-1).tolist())
                # 写入标注信息
                f.write(f"{class_ids[i]} " + " ".join(boxes_i) + "\n")