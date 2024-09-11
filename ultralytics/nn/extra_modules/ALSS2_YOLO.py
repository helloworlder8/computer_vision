import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = (
    "MSCAM",
    "SELayer",
    "SaELayer",
    "MSCAM",
)


# original_repr = torch.Tensor.__repr__
# # 定义自定义的 __repr__ 方法
# def custom_repr(self):
#     return f'{self.shape} {original_repr(self)}'
#     # return f'{self.shape}'
# # 替换 torch.Tensor 的 __repr__ 方法
# torch.Tensor.__repr__ = custom_repr


# 定义SE模块
class SELayer(nn.Module):
    def __init__(self, c1, c_=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // c_, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // c_, c1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x): #torch.Size([8, 64, 32, 32])
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #torch.Size([8, 64])
        y = self.fc(y).view(b, c, 1, 1) #torch.Size([8, 64, 1, 1])
        return x * y.expand_as(x)

# 定义SaE模块
class SaELayer(nn.Module):
    def __init__(self, c1, c_=4):
        super(SaELayer, self).__init__()
        self.c_ = c_
        self.cardinality = 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # cardinality 1
        self.fc1 = nn.Sequential(
            nn.Linear(c1, c_, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 2
        self.fc2 = nn.Sequential(
            nn.Linear(c1, c_, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 3
        self.fc3 = nn.Sequential(
            nn.Linear(c1, c_, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 4
        self.fc4 = nn.Sequential(
            nn.Linear(c1, c_, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(c_ * self.cardinality, c1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x): #torch.Size([8, 128, 32, 32])
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #torch.Size([8, 128])
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y_concate = torch.cat([y1, y2, y3, y4], dim=1) #torch.Size([8, 16])
        y_ex_dim = self.fc(y_concate).view(b, c, 1, 1)

        return x * y_ex_dim.expand_as(x) #torch.Size([8, 128, 32, 32]) torch.Size([8, 128, 1, 1])


class MSCAM(nn.Module):
    def __init__(self, c1, c_=4):
        super(MSCAM, self).__init__()
        self.c_ = c_
        self.cardinality = 4
        
        self.channel_reduction = nn.Conv2d(c1, c_, kernel_size=3, bias=False)
        
        self.conv1 = nn.Conv2d(c_, c_, kernel_size=1, groups=c_, bias=False)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(c_, c_, kernel_size=3, groups=c_, bias=False)
        self.pool2 = nn.AdaptiveAvgPool2d(3)
        self.conv3 = nn.Conv2d(c_, c_, kernel_size=5, groups=c_, bias=False)
        self.pool3 = nn.AdaptiveAvgPool2d(5)
        self.conv4 = nn.Conv2d(c_, c_, kernel_size=7, groups=c_, bias=False)
        self.pool4 = nn.AdaptiveAvgPool2d(7)
        
        self.channel_expansion = nn.Conv2d(c_ * self.cardinality, c1, kernel_size=1, bias=False)
        self.act = nn.SiLU()
        self.Sig = nn.Sigmoid()

    def forward(self, x):
        y = self.act(self.channel_reduction(x))

        y1 = self.pool1(y)
        y1 = self.conv1(y1)
        y1 = self.act(y1)

        y2 = self.pool2(y)
        y2 = self.conv2(y2)
        y2 = self.act(y2)

        y3 = self.pool3(y)
        y3 = self.conv3(y3)
        y3 = self.act(y3)
        
        y4 = self.pool4(y)
        y4 = self.conv4(y4)
        y4 = self.act(y4)

        y_concat = torch.cat([y1, y2, y3, y4], dim=1)
        
        y = self.Sig(self.channel_expansion(y_concat))
        y = y.expand_as(x)
        return x * y
