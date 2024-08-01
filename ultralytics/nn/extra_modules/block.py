

import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv

__all__ = (
    "CoordAttv2",
    "ASSNET",
)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x

def channel_shuffle(x, g):
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // g
    # Reshape
    x = x.view(batch_size, g, channels_per_group, height, width)
    # Transpose 1 and 2 axis
    x = x.transpose(1, 2).contiguous()
    # Flatten
    x = x.view(batch_size, -1, height, width)
    return x


    
class ASSNET(nn.Module):
    #              为了保证程序可靠运行并不添加程序复杂度   请保证in_channels*  split_ratio 可以被left_conv_group整除
    def __init__(self, in_channels, out_channels, n=1, split_ratio=0.2, stride=1, conv_or_identity=0,conv_poolconv_pool=0,right_bottleneckratio=1): 
        super(ASSNET, self).__init__()                                          
        self.in_left=int(in_channels*split_ratio)
        self.in_right=in_channels-self.in_left
        self.in_right_mid = int(self.in_right*right_bottleneckratio)
        # self.in_right_mid = 1
        self.out_right=  out_channels-self.in_left
        self.n=n


        assert stride in [1, 2]
        if stride == 1:
            if conv_or_identity==0:
                self.shortcut = Conv(self.in_left, self.in_left, 3, 1)
            else:
                self.shortcut = Identity()
            self.m = nn.ModuleList(Conv(self.in_right_mid, self.in_right_mid, 3, 1, g=self.in_right_mid,act=False)for _ in range(n))



        if stride == 2:
            # 步长为1 前期卷积  后期恒等连接      步长为2 卷积 池化卷积  后期池化
            if conv_poolconv_pool==0:
                self.shortcut = Conv(self.in_left, self.in_left, 3, 2)
            if conv_poolconv_pool==1:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    Conv(self.in_left, self.in_left, 3, 1)
                )
            else:
                self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)          
            # self.cv2 = Conv(self.in_right_mid, self.in_right_mid, 3, 2, g=self.in_right_mid,act=False)  # Depthwise convolution
            self.m = nn.ModuleList()
            self.m.append(Conv(self.in_right_mid, self.in_right_mid, 3, 2, g=self.in_right_mid,act=False))
            for _ in range(1, n):
                self.m.append(Conv(self.in_right_mid, self.in_right_mid, 3, 1, g=self.in_right_mid,act=False))
            # self.m = nn.ModuleList(Conv(self.in_right_mid, self.in_right_mid, 3, 2, g=self.in_right_mid,act=False)for _ in range(n))


        self.cv1 = Conv(self.in_right, self.in_right_mid, 3,1)

        # self.cv3 = Conv(self.in_right_mid, self.in_right_mid, 3, 1, g=int(self.in_right_mid//2))  # Depthwise convolution
        self.cv3 = Conv(self.in_right_mid, self.out_right, 3,1)



    def forward(self, x):
        proportional_sizes=[self.in_left,self.in_right]
        x = list(x.split(proportional_sizes, dim=1))
        x_left = self.shortcut(x[0])
        x_right = self.cv1(x[1])
        for module in self.m:
            x_right = module(x_right)
        x_right = self.cv3(x_right) #torch.Size([1, 103, 40, 40]) torch.Size([2, 30, 160, 160])
        x= torch.cat((x_right, x_left), dim=1)
        x = channel_shuffle(x, 2)
        return x
    
    

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
    
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)
    
class CoordAttv2(nn.Module):
    def __init__(self, input_channel, reduction=32):
        super(CoordAttv2, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 

 
        self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.act = h_swish()
 
        self.conv_h = nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, padding=0,groups=input_channel)
        self.conv_w = nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, padding=0,groups=input_channel)

    
    def forward(self, x):  # torch.Size([2, 32, 64, 64])
        identity = x

        b, c, h, w = x.size()
        x_h = self.pool_h(x)  # torch.Size([2, 32, 64, 1])
        x_w = self.pool_w(x)  # torch.Size([2, 32, 1, 64]) 



        a_h = self.conv_h(x_h).sigmoid()  #  torch.Size([2, 32, 64, 1])
        a_w = self.conv_w(x_w).sigmoid()  #  torch.Size([2, 32, 1, 64])

        out = identity * a_w * a_h  # torch.Size([2, 32, 64, 64])

        return out
    
    
# 测试 CoordAtt 模块
if __name__ == '__main__':
    torch.manual_seed(0)
    input_tensor = torch.rand(2, 32, 64, 64)  # 假设输入尺寸为 [batch_size, channels, height, width]
    coord_att = CoordAttv2(input_channel=32)  # 创建 CoordAtt 实例，假设输入通道为32
    output = coord_att(input_tensor)  # 获取输出

    print(f'Input shape: {input_tensor.shape}')
    print(f'Output shape: {output.shape}')

    # 如果需要，还可以使用 torchsummary 来查看模型的详细信息
    # summary(coord_att, input_size=(32, 64, 64))