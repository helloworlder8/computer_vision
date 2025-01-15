import torch
original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr



import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "BSCA",
)




class StochasticPooling(nn.Module):
    def forward(self, x):
        # Apply random noise
        rand_weight = torch.rand_like(x).to(x.device)
        x_weighted = x * rand_weight
        
        # Global pooling
        batch_size, num_channels, height, width = x.size()
        stochastic_pooled = F.avg_pool2d(x_weighted, (height, width))
        
        return stochastic_pooled



class BSCA(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(BSCA, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.stochastic_pool = StochasticPooling()

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // ratio, in_channels, bias=False)
        )

    def forward(self, x):
        # Batch Normalization
        x_bn = self.bn(x)

        # Different Pooling Mechanisms
        max_pool_out = self.max_pool(x_bn).view(x_bn.size(0), -1)
        avg_pool_out = self.avg_pool(x_bn).view(x_bn.size(0), -1)
        stochastic_pool_out = self.stochastic_pool(x_bn).view(x_bn.size(0), -1)

        # Shared MLP
        max_pool_out = self.mlp(max_pool_out)
        avg_pool_out = self.mlp(avg_pool_out)
        stochastic_pool_out = self.mlp(stochastic_pool_out)

        # Summing Up
        combined_out = max_pool_out + avg_pool_out + stochastic_pool_out

        # Sigmoid Activation
        attention_map = torch.sigmoid(combined_out).unsqueeze(2).unsqueeze(3)

        return x * attention_map

# Example usage
if __name__ == '__main__':
    input_tensor = torch.randn(1, 64, 128, 128)  # Batch size of 1, 64 channels, 128x128 feature map
    channel_attention_module = BSCA(in_channels=64)
    output_tensor = channel_attention_module(input_tensor)
    print(output_tensor.shape)  # Should be torch.Size([1, 64, 128, 128])
