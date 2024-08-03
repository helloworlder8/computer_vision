import torch

def load_model_and_print_epochs(model_path):
    # 加载模型文件
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # 检查'epochs'键是否存在，并打印其值
    if 'epoch' in checkpoint:
        print(f"Epochs in checkpoint: {checkpoint['epoch']}")
    else:
        print("The key 'epochs' does not exist in this checkpoint.")

# 调用函数，传入模型文件路径
model_path = 'runs/detect/train/weights/best.pt'
load_model_and_print_epochs(model_path)
