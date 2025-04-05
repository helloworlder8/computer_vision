import torch

# 定义模型路径
model_path = 'yolov8n.pt'

# 加载模型
model = torch.load(model_path, map_location=torch.device('cpu'))

# 即便官方模型的许多参数都用不了
# 打印模型的键
if isinstance(model, dict):
    print("Model keys:", model.keys())
else:
    print("Loaded object is not a dictionary. It might be a model object.")


# 时间 版本 开源协议 文档 当前训练轮次 最好拟合值 模型 ema update 优化器 训练参数（这里改了好多） 训练指标（最后的） 训练结果（每次的） 
# ['date', 'version', 'license', 'docs', 'epoch', 'best_fitness', 'model', 'ema', 'updates', 'optimizer', 'train_args', 'train_metrics', 'train_results']