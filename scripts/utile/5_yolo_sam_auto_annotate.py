from pathlib import Path
from ultralytics import SAM, YOLO
import torch

# 定义图像数据路径
img_str = 'ultralytics/assets'

# 定义检测模型和SAM模型的路径
det_model_pt="yolov8n.pt"
sam_model_pt="sam_b.pt"

# 根据CUDA是否可用选择设备
device = '0' if torch.cuda.is_available() else 'cpu'

# 定义输出目录，默认为None
# 输出路径
output_path = None

# 初始化检测模型和SAM模型
det_model = YOLO(det_model_pt)  
sam_model = SAM(sam_model_pt)

# 获取图像数据路径
img_path = Path(img_str)

# 如果输出目录未定义，则生成默认的输出目录
if not output_path:
    output_path = img_path.parent / f"{img_path.stem}_auto_annotate_labels"
    # 创建输出目录
    Path(output_path).mkdir(exist_ok=True, parents=True)

# 对图像数据进行检测
det_results = det_model(img_path, stream=True, device=device)

# 遍历检测结果
for result in det_results:
    # 获取类别ID
    class_ids = result.boxes.cls.int().tolist()  # noqa
    # 如果有检测到物体
    if len(class_ids):
        # 获取检测框坐标
        boxes = result.boxes.xyxy  # Boxes object for bbox outputs  #torch.Size([6, 4])
        # 使用SAM模型进行分割
        sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False, device=device)
        # 获取分割结果
        segments = sam_results[0].masks.xyn  # noqa
        # 为每个图像生成标注文件
        with open(f"{Path(output_path) / Path(result.path).stem}.txt", "w") as f:
            # 遍历每个分割区域
            for i in range(len(segments)):
                s = segments[i]
                # 如果分割区域为空，则跳过
                if len(s) == 0:
                    continue
                # 将分割区域坐标转换为字符串格式
                segment = map(str, segments[i].reshape(-1).tolist())
                # 写入标注信息
                f.write(f"{class_ids[i]} " + " ".join(segment) + "\n")
