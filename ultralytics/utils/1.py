import torch
import math

# 设置 sin_best 的值
sin_best = math.pi / 2

# 计算 torch.arcsin(sin_best) 的值
angle = torch.arcsin(torch.tensor(sin_best))

# 计算最终表达式的值
result = torch.cos(torch.tensor(1.746) - math.pi / 2)

# 输出结果
print("The result is:", result.item())
# 0.1737