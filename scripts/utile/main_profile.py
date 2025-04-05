  
import os
import torch
from tqdm import tqdm
from ultralytics.nn.tasks import DetectionModel,create_model_dict


def process_model(model_yaml,ch):
    # 创建模型字典
    # 先打印一个空行确保换行
    print(f"\n\n\n\n")
    # 然后打印模型路径
    # print(f"\033[1mProcessing model: {model_yaml}\033[0m")
    model_dict = create_model_dict(model_yaml)
    # 创建检测模型
    detection_model = DetectionModel(model_dict, ch=ch, nc=8, verbose=True)
    # 融合模型（如果适用）DetectionModel
    print(f"\033[1mProcessing model: {model_yaml}\033[0m")
    detection_model.fuse()
    # 创建一个输入张量
    x = torch.zeros(1, ch, 640, 640)  # 注意检查通道数是否正确
    # 执行一次预测
    detection_model._predict_once(x)

def process_all_models(directory,ch):
    # 列出给定目录下的所有.yaml文件
    files = [f for f in os.listdir(directory) if f.endswith(".yaml")]
    for filename in tqdm(files, desc="Processing Models", unit="model"):
        full_path = os.path.join(directory, filename)
        process_model(full_path,ch)


if __name__ == '__main__':
    # model_directory = "ultralytics/cfg_yaml/models/ALSS2/ALSSm-seg.yaml"
    model_directory = "ultralytics/cfg_yaml/models/ALSS-YOLO-world/ALSS-YOLO-world.yaml"
    ch = 3
    # 检查model_directory是文件还是文件夹
    if '.' in os.path.basename(model_directory) and model_directory.rsplit('.', 1)[-1]:
        process_model(model_directory, ch)
    elif os.path.isdir(model_directory):
        process_all_models(model_directory, ch)
    else:
        print(f"Error: {model_directory} is neither a valid file with a suffix nor a directory.")
# ultralytics/cfg_yaml/models/v8/yolov8-ghost.yaml

#yolov3s.yaml 10,256,616
#yolov5s.yaml 9,114,632
#yolov6s.yaml 10,920,488
#yolov8s.yaml 11,128,680
#yolov9s.yaml 7,170,184
#yolov10s.yaml 7,989,936 24.5
#yolov11s.yaml 10,713,816
#yolov8s-p2.yaml 10,093,448
#yolov8m-ghost.yaml 10,279,74
#yolov8m-ghost-p2.yaml 8,988,064



# yolov8-rtdetr.yaml

# 2,904,992  v11n