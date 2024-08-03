import torch

# 加载 .pt 文件
file_path = 'runs/detect/V8-/weights/last.pt'
data = torch.load(file_path)

# 检查 'Detection_Model' 键是否存在
if 'Detection_Model' in data:
    # 检查 'start_flops_count' 是否存在于 'Detection_Model' 字典中
    if 'start_flops_count' not in data['Detection_Model']:
        data['Detection_Model']['start_flops_count'] = 0  # 添加 'start_flops_count' 并赋值为0
    
    # 保存修改后的数据回 .pt 文件
    torch.save(data, file_path)
    print("File updated successfully with start_flops_count set to 0.")
else:
    print("Key 'Detection_Model' not found in the file.")
