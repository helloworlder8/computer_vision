import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs): #128 10
        super(Attention, self).__init__(**kwargs)
        
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.bias = bias

        self.weights = nn.Parameter(torch.Tensor(feature_dim, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)

        # 初始化权重
        nn.init.kaiming_uniform_(self.weights, a=0, mode='fan_in', nonlinearity='leaky_relu')
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x):
        # x shape: (batch_size, time_steps, features) #torch.Size([2, 10, 128])  torch.Size([128, 1])
        u = torch.tanh(torch.matmul(x, self.weights) + self.bias) #->torch.Size([2, 10, 1])

        # 转换成概率分布（权重）
        attn = torch.softmax(u, dim=1) #torch.Size([2, 10, 1])

        # 加权求和
        weighted_output = torch.mul(x, attn) ##torch.Size([2, 10, 128])   torch.Size([2, 10, 128])

        # 求和所有时间步
        outputs = torch.sum(weighted_output, dim=1) #torch.Size([2, 128])
        return outputs, attn

# 定义模型参数
feature_dim = 128  # 特征的维度
step_dim = 10      # 时间步长度

# 创建注意力层实例
attention_layer = Attention(feature_dim, step_dim)

# 假设有一个简单的输入 batch_size=32, step_dim=10, feature_dim=128
x = torch.randn(2, 10, 128)

# 前向传播
output, attn_weights = attention_layer(x)
print("Output shape:", output.shape)  # 输出形状应为 (batch_size, feature_dim)
print("Attention Weights shape:", attn_weights.shape)  # 注意力权重形状应为 (batch_size, time_steps, 1)
