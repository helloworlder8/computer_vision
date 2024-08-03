import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class IdentityBasedConv1x1(nn.Module):
    def __init__(self, channels, groups=1):
        super().__init__()
        assert channels % groups == 0 #可以被整除
        input_dim = channels // groups #4
        self.conv = nn.Conv2d(channels,channels, 1, groups=groups, bias=False) #1x1分组卷积
        # 卷积核尺寸乘组数=输出通道数
        id_value = np.zeros((channels, input_dim, 1, 1)) #(8, 4, 1, 1)
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = torch.from_numpy(id_value).float() 
        nn.init.zeros_(self.conv.weight)
        self.groups = groups
    
    def forward(self, input): #torch.Size([1, 8, 4, 4])
        kernel = self.conv.weight + self.id_tensor.to(self.conv.weight.device).type_as(self.conv.weight) #torch.Size([8, 4, 1, 1])
        result = F.conv2d(input, kernel, None, stride=1, groups=self.groups)
        return result

    def get_actual_kernel(self):
        return self.conv.weight + self.id_tensor.to(self.conv.weight.device).type_as(self.conv.weight)

# 测试IdentityBasedConv1x1
channels = 8
groups = 2
model = IdentityBasedConv1x1(channels, groups)
input_tensor = torch.randn(1, channels, 4, 4)  # 创建一个随机输入张量

output = model(input_tensor)
print("Output shape:", output.shape)
