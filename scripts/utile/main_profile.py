  
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
    detection_model = DetectionModel(model_dict, ch=ch, nc=65, verbose=True)
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
    model_directory = "runs/ISOD-datasets"
    ch = 1
    # 检查model_directory是文件还是文件夹
    if os.path.isfile(model_directory):
        process_model(model_directory,ch)
    elif os.path.isdir(model_directory):
        process_all_models(model_directory,ch)
    else:
        print(f"Error: {model_directory} is neither a file nor a directory.")
        
# Model summary (fused): 173 layers, 2,763,087 parameters, 2,763,071 gradients''
# Model summary (fused): 173 layers, 2,740,479 parameters, 2,740,463 gradients