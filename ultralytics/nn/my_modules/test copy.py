import math

import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x))) #torch.Size([2, 16, 64, 64]) torch.Size([2, 16, 1, 1])
    


# 单元测试函数
def test_channel_attention():
    torch.manual_seed(0)  # 设置随机种子以保证结果可复现
    

    batch_size = 2
    channels = 16
    height, width = 64, 64
    
    # 创建 ChannelAttention 实例
    attention_module = ChannelAttention(channels)
    
    # 创建一个随机输入张量
    inputs = torch.randn(batch_size, channels, height, width)
    
    # 运行 forward 传递
    outputs = attention_module(inputs)
    
    # 检查输出形状是否正确
    assert outputs.shape == inputs.shape, "Output shape is not equal to input shape"
    
    # 检查是否有任何变化（测试是否有激活应用）
    assert not torch.allclose(inputs, outputs), "Output is too close to input, check attention effectiveness"

    print("Test passed successfully. Output shape:", outputs.shape)

# 调用测试函数
test_channel_attention()