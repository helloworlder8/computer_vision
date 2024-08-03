import os
import torch
from tqdm import tqdm
from ultralytics.nn.tasks_model import Detection_Model,creat_model_dict_add



def process_model(model_path):
    # 创建模型字典
    # 先打印一个空行确保换行
    print(f"\n\n\n\n")
    # 然后打印模型路径
    # print(f"\033[1mProcessing model: {model_path}\033[0m")
    model_dict = creat_model_dict_add(model_path)
    # 创建检测模型
    detection_model = Detection_Model(model_dict, ch=1, nc=4, verbose=True, ptflops=False)
    # 融合模型（如果适用）
    print(f"\033[1mProcessing model: {model_path}\033[0m")
    detection_model.fuse()
    # 创建一个输入张量
    x = torch.zeros(1, 1, 640, 640)  # 注意检查通道数是否正确
    # 执行一次预测
    detection_model._predict_once(x)

def process_all_models(directory):
    # 列出给定目录下的所有.yaml文件
    files = [f for f in os.listdir(directory) if f.endswith(".yaml")]
    for filename in tqdm(files, desc="Processing Models", unit="model"):
        full_path = os.path.join(directory, filename)
        process_model(full_path)

# 设定你的模型配置文件所在的文件夹路径
model_directory = "runs/ablation_experiment/ass-focus-poolconv-atta-iou-0.891/ASS-newfocus-poolconv-atta.yaml"
# process_all_models(model_directory)
process_model(model_directory)