

import math

import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv
from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init

__all__ = (
    "GSConv",
    "VoVGSCSP", 
    "VoVGSCSPC",
    "PConv",
    "ShuffleNetV2",
    "space_to_depth",
    "ShuffleNetV3",
    "CoordAtt",
    "ContextAggregation",
    "AFF",
    "iAFF",
    "CoordAttv2",
    "ASSNET",
)

######################   slim-neck-by-gsconv ####     start ###############################

class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2 #
        self.cv1 = Conv(c1, c_, k, s, None, g, 1,  act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, 1 , act)
 
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])
 
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)
 
        return torch.cat((y[0], y[1]), 1)
 
class GSConvns(GSConv):
    # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__(c1, c2, k=1, s=1, g=1, act=True)
        c_ = c2 // 2
        self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)
 
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # normative-shuffle, TRT supported
        return nn.ReLU(self.shuf(x2))
 
 
class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2*e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)
 
    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)
 
 
class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, g
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)
 
 
class GSBottleneckC(GSBottleneck):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, k, s, act=False)
 
 
class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        # self.gc1 = GSConv(c_, c_, 1, 1)
        # self.gc2 = GSConv(c_, c_, 1, 1)
        # self.gsb = GSBottleneck(c_, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)  #
 
 
    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))
 
 
class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2)
        c_ = int(c2 * 0.5)  # hidden channels
        self.gsb = GSBottleneckC(c_, c_, 1, 1)
 

######################   slim-neck-by-gsconv ####     end ###############################



###################### PConv  ####     start############################# 效果好
class PConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        self.conv = Conv(dim, ouc, k=1) #拼完之后这个卷积
 
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
 
    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        x = self.conv(x)
        return x
 
    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv(x)
        return x
 
 
###################### PConv  ####     end#############################
    
class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act) #1x1卷积
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act) #5x5卷积通道可分离

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection.""" 
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1) #两条支路拼接




###################### ShuffleNetV2  ####     start#############################
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

class ShuffleNetV2(nn.Module):
    def __init__(self, c1, c2, s, n=1, e=0.5, pool=False): #输入 输出 步长 内部多少个 类似botleneck结构 池化
        super(ShuffleNetV2, self).__init__()
        self.n = n
        self.s = s
        self.pool = pool
        c_ = int(c2//2 * e)
        c__ = int(c_ * e)

        assert s in [1, 2]

        if s == 2:
            if self.pool:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    Conv(c1, c2//2, 3, 1, 1)
                )
            else:
                self.shortcut = Conv(c1, c2//2, 3, 2)
        else:
            self.shortcut = Conv(c1, c2//2, 3, 1)

        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_, c_, 3, s, 1, g=c_)  # Depthwise convolution
        self.cv_2_1 = Conv(c_, c__, 1)
        self.cv_2_2 = Conv(c__, c__, 3, 1, 1, g=c__)
        self.cv_2_3 = Conv(c__, c_, 1)
        self.c3 = Conv(c_, c2//2, 1)

    def forward(self, x):

        residual = self.shortcut(x)

        x = self.cv1(x)
        x = self.cv2(x)
        if self.n == 2:
            x = self.cv_2_3(self.cv_2_2(self.cv_2_1(x))) #torch.Size([1, 2, 32, 32]) torch.Size([1, 1, 32, 32])
        x = self.c3(x)

        # if self.left_part:
        out = torch.cat((residual, x), 1)


        out = channel_shuffle(out, 2)
        return out
###################### ShuffleNetV2  ####     end#############################

 



    









###################### ShuffleNetV3  ####     start#############################
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x
    
def adjust_groups(in_channels, desired_groups):
    """
    调整分组数以确保输入通道数能被分组数整除。
    如果初始分组数不能整除输入通道数，向下调整分组数找到最大的临近值。
    """
    while in_channels % desired_groups != 0:
        desired_groups -= 1
    return desired_groups

class ShuffleNetV3(nn.Module):
    #              为了保证程序可靠运行并不添加程序复杂度   请保证in_channels*  split_ratio 可以被left_conv_group整除
    def __init__(self, in_channels, out_channels, split_ratio=0.2, stride=1, conv_or_identity=0,conv_poolconv_pool=0,desired_groups=1,right_bottleneckratio=1): 
        super(ShuffleNetV3, self).__init__()                                          
        self.in_left=int(in_channels*split_ratio)
        self.in_right=in_channels-self.in_left
        self.in_right_mid = int(self.in_right*right_bottleneckratio)
        # self.in_right_mid = 1
        self.out_right=  out_channels-self.in_left
        desired_groups = int(self.in_left*desired_groups)
        self.groups = adjust_groups(self.in_left, desired_groups)

        assert stride in [1, 2]
        if stride == 1:
            if conv_or_identity==0:
                self.shortcut = Conv(self.in_left, self.in_left, 3, 1,g=self.groups)
            else:
                self.shortcut = Identity()
            self.cv2 = Conv(self.in_right_mid, self.in_right_mid, 3, 1, g=self.in_right_mid,act=False)  # Depthwise convolution




        if stride == 2:
            # 步长为1 前期卷积  后期恒等连接      步长为2 卷积 池化卷积  后期池化
            if conv_poolconv_pool==0:
                self.shortcut = Conv(self.in_left, self.in_left, 3, 2, g=self.groups)
            if conv_poolconv_pool==1:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    Conv(self.in_left, self.in_left, 3, 1,g=self.groups)
                )
            else:
                self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)          
            self.cv2 = Conv(self.in_right_mid, self.in_right_mid, 3, 2, g=self.in_right_mid,act=False)  # Depthwise convolution



        self.cv1 = Conv(self.in_right, self.in_right_mid, 1,1)

        # self.cv3 = Conv(self.in_right_mid, self.in_right_mid, 3, 1, g=int(self.in_right_mid//2))  # Depthwise convolution
        self.cv3 = Conv(self.in_right_mid, self.out_right, 1,1)



    def forward(self, x):
        proportional_sizes=[self.in_left,self.in_right]
        x = list(x.split(proportional_sizes, dim=1))
        x_left = self.shortcut(x[0])
        x_right = self.cv3(self.cv2(self.cv1(x[1]))) #torch.Size([1, 103, 40, 40])
        x= torch.cat((x_right, x_left), dim=1)
        x = channel_shuffle(x, 2)
        return x
# , n=1
###################### ShuffleNetV2  ####     end#############################
    
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




class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act) #1x1卷积
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act) #5x5卷积通道可分离

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection.""" 
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1) #两条支路拼接
class ASSNET2(nn.Module):
    #              为了保证程序可靠运行并不添加程序复杂度   请保证in_channels*  split_ratio 可以被left_conv_group整除
    def __init__(self, in_channels, out_channels, n_div=4): 
        super(ASSNET2, self).__init__()                                          
        self.in_left=in_channels// n_div
        self.in_right=in_channels-self.in_left
  
        # self.in_right_mid = 1
        self.out_right=  out_channels-self.in_left


        # self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

        self.shortcut = Identity()




        self.cv1 = Conv(self.in_right, self.in_right, 5,1,2,self.in_right)

        # self.cv3 = Conv(self.in_right_mid, self.in_right_mid, 3, 1, g=int(self.in_right_mid//2))  # Depthwise convolution
        self.cv3 = Conv(self.in_right, self.out_right, 1,1)



    def forward(self, x):
        x1, x2 = torch.split(x, [self.in_left, self.in_right], dim=1)
        x_right = self.cv1(x2)
        x_right = self.cv3(x_right) #torch.Size([1, 103, 40, 40]) torch.Size([2, 30, 160, 160])
        x= torch.cat((x1, x_right), dim=1)
        # x = channel_shuffle(x, 2)
        return x

# class C2f(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1) for _ in range(n))

#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))









# ###################### ShuffleNetV2  ####     end#############################
# class Bottleneck(nn.Module):
#     """Standard bottleneck."""

#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
#         """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
#         expansion.
#         """
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g) #分组卷积
#         self.add = shortcut and c1 == c2  #相等并且输入等于输出

#     def forward(self, x):
#         """'forward()' applies the YOLO FPN to input data."""
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

# class C2f(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):#e控制中间通道数
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=0.5) for _ in range(n))#e=1.0就不是bottlneck
#         # false直接卷积不要残差连接
#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1)) #宽高不变      一个卷积直接过来
#         y.extend(m(y[-1]) for m in self.m) # 所有步长全为1
#         return self.cv2(torch.cat(y, 1)) #通道拼接

#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))
###################### ShuffleNetV2  ####     end#############################













######################  SPD-Conv  ####     start ###############################
 
class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
 
    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
 
######################  SPD-Conv  ####     start ###############################
    




######################  CoordAtt  ####     start   by  AI&CV  ###############################
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
 
 

######################  CoordAtt  ####     end   by  AI&CV  ###############################


class ContextAggregation(nn.Module):  #太差了
    """
    Context Aggregation Block.
    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """
    
    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels  #8
        self.reduction = reduction      #2
        self.inter_channels = max(in_channels // reduction, 1) #4
    
        conv_params = dict(kernel_size=1, act_cfg=None) #配置卷积核和激活
    
        self.conv1 = ConvModule(in_channels, 1, **conv_params) #8 1 1 1
        self.conv2 = ConvModule(in_channels, 1, **conv_params) #8 1 1 1
        self.conv3 = ConvModule(in_channels, self.inter_channels, **conv_params) #8 4 1 1
        self.conv4 = ConvModule(self.inter_channels, in_channels, **conv_params) #4 8 1 1
    
        self.init_weights()
    
    def init_weights(self):
        for m in (self.conv1, self.conv2, self.conv3):
            caffe2_xavier_init(m.conv) #Xavier 初始化，也称为 Glorot 初始化，是一种在神经网络中初始化权重的方法。它旨在使得每一层的输入和输出的方差尽可能相等，从而有助于梯度的稳定传播，特别是在网络的训练初期。
        constant_init(self.conv4.conv, 0)
    # constant_init 函数的作用是将给定的卷积层（self.m.conv）的权重（包括权重张量和可能的偏置项）设置为常数值，这里指定的常数值为 0。这意味着该卷积层的所有权重在初始化后将为零。
    def forward(self, x):
        #batch_size, inter_channels = x.size(0)
        batch_size = x.size(0)
        inter_channels = self.inter_channels #4
        #batch_size, nH, nW, inter_channels = x.shape
    
        # x1: [batch_size, 1, H, W]
        x1 = self.conv1(x).sigmoid() #torch.Size([1, 1, 16, 16]) 通道降低到1 0-1之间
    
        # x2: [batch_size, 1, HW, 1]  
        x2 = self.conv2(x).view(batch_size, 1, -1, 1).softmax(2) #torch.Size([2, 1, 256, 1])
    
        # x3: [batch_size, 1, inter_channels, HW] 
        x3 = self.conv3(x).view(batch_size, 1, inter_channels, -1) # 2 4 16 16->2 1 4 256
    
        # y: [batch_size, inter_channels, 1, 1]   #torch.Size([2, 1, 4, 1])
        x4 = torch.matmul(x3, x2).view(batch_size, inter_channels, 1, 1)
        y = self.conv4(x4) * x1 # 2 4 1 1 -> 2 8 1 1     1 1 16 16
    
        return x + y
    
##################### ContextAggregation  ####     END   by  AI&CV  ###############################



# from random import weibullvariate

 
class AFF(nn.Module):
	"""
	Implimenting AFF module
	"""
 
	def __init__(self, channels=64, r=4):
		super(AFF, self).__init__()
		inter_channels = int(channels //r )
 
		self.local_att = nn.Sequential(
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)
 
		self.global_att = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)
 
		self.sigmoid = nn.Sigmoid()
 
	def forward(self, input):
		x= input[0]
		y= input[1]
		xa = x + y
		xl = self.local_att(xa)
		xg= self.global_att(xa)
		xlg = xl + xg
		m = self.sigmoid(xlg)
 
		x_union_y = 2* x * m + 2* y * (1-m)
 
		return x_union_y
 
 
 
class iAFF(nn.Module):
 
	"""
	implimenting iAFF module
	"""
 
	def __init__(self, channels=64, r=4):
		super(iAFF, self).__init__()
		inter_channels = int(channels // r)
 
		self.local_attention1 = nn.Sequential(
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)
		self.global_attention1 = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)
 
		self.local_attention2 = nn.Sequential(
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)
		self.global_attention2 = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(inter_channels),
		#	nn.ReLU(inplace=True),
                        nn.SiLU(),
			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
			nn.BatchNorm2d(channels),
		)
 
		self.sigmoid = nn.Sigmoid()
 
 
	def forward(self, input):
		"""
		Implimenting the iAFF forward step
		"""
 
		x = input[0]
		y = input[1]
		xa = x+y
		xl = self.local_attention1(xa)
		xg = self.global_attention1(xa)
		xlg = xl+xg
		m1 = self.sigmoid(xlg)
		xuniony = x * m1 + y * (1-m1)
 
		xl2 = self.local_attention2(xuniony)
		xg2 = self.global_attention2(xuniony)
		xlg2 = xl2 + xg2
		m2 = self.sigmoid(xlg2)
		z = x * m2 + y * (1-m2)
		return z
 

class CoordAtt(nn.Module):
    def __init__(self, input_channel, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        mip = max(8, input_channel // reduction)
 
        self.conv1 = nn.Conv2d(input_channel, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
 
        self.conv_h = nn.Conv2d(mip, input_channel, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, input_channel, kernel_size=1, stride=1, padding=0)
 
    def forward(self, x):
        identity = x
 
        b, c, h, w = x.size()
        x_h = self.pool_h(x) #torch.Size([2, 32, 64, 1])
        x_w = self.pool_w(x).permute(0, 1, 3, 2) #torch.Size([2, 32, 64, 1])
 
        y = torch.cat([x_h, x_w], dim=2) #torch.Size([2, 32, 128, 1])
        y = self.conv1(y) #torch.Size([2, 8, 128, 1])
        y = self.bn1(y)
        y = self.act(y)
 
        x_h, x_w = torch.split(y, [h, w], dim=2) #torch.Size([2, 8, 64, 1]) torch.Size([2, 8, 64, 1])
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return out

def feature_shuffle(x, g):
    batch_size, num_channels, height, width = x.data.size()
    height_per_group = height // g
    # Reshape
    x = x.view(batch_size, num_channels, g, height_per_group, width)
    # Transpose 1 and 2 axis
    x = x.transpose(2, 3).contiguous()
    # Flatten
    x = x.view(batch_size, num_channels, -1, width)
    return x

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
    coord_att = CoordAtt(input_channel=32)  # 创建 CoordAtt 实例，假设输入通道为32
    output = coord_att(input_tensor)  # 获取输出

    print(f'Input shape: {input_tensor.shape}')
    print(f'Output shape: {output.shape}')

    # 如果需要，还可以使用 torchsummary 来查看模型的详细信息
    # summary(coord_att, input_size=(32, 64, 64))