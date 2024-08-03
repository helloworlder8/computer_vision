import torch

# 加载模型
model_path = 'runs/weight_contrast_experiment/V8-0.895/weights/best.pt'
model = torch.load(model_path)

# 检查模型中是否有 'train_args' 键
if 'train_args' in model and 'device' in model['train_args']:
    # 从 'train_args' 中删除 'device' 键
    del model['train_args']['device']
    # 保存修改后的模型
    torch.save(model, model_path)
    print("已从 'train_args' 中删除 'device' 并保存了修改后的模型。")
else:
    print("'train_args' 不存在或 'device' 在 'train_args' 中不存在，无需修改。")
