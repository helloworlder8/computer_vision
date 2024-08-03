


import math

import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv


# ###################### ContextAggregation  ####     START   by  AI&CV  ###############################
    


from mmcv.cnn import ConvModule
from mmengine.model import caffe2_xavier_init, constant_init
class ContextAggregation(nn.Module):
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



# import torch
# import torch.nn as nn
# from mmcv.cnn import DepthwiseSeparableConvModule
# from mmengine.model import caffe2_xavier_init, constant_init

# class ContextAggregation(nn.Module):
#     def __init__(self, in_channels, reduction=1):
#         super(ContextAggregation, self).__init__()
#         self.in_channels = in_channels
#         self.reduction = reduction
#         self.inter_channels = max(in_channels // reduction, 1)
        
#         # Use depthwise separable convolutions
#         conv_params = dict(kernel_size=1, act_cfg=None)
        
#         self.conv1 = DepthwiseSeparableConvModule(in_channels, 1, **conv_params)
#         self.conv2 = DepthwiseSeparableConvModule(in_channels, 1, **conv_params)
#         self.conv3 = DepthwiseSeparableConvModule(in_channels, self.inter_channels, **conv_params)
#         self.conv4 = DepthwiseSeparableConvModule(self.inter_channels, in_channels, **conv_params)
        
#         self.init_weights()
        
#     def init_weights(self):
#         for m in (self.conv1, self.conv2, self.conv3):
#             caffe2_xavier_init(m.depthwise_conv)
#             caffe2_xavier_init(m.pointwise_conv)
#         constant_init(self.conv4.depthwise_conv, 0)
#         constant_init(self.conv4.pointwise_conv, 0)

#     def forward(self, x):
#         batch_size = x.size(0)
#         inter_channels = self.inter_channels
        
#         x1 = self.conv1(x).sigmoid()
#         x2 = self.conv2(x).view(batch_size, 1, -1, 1).softmax(dim=2)
#         x3 = self.conv3(x).view(batch_size, 1, inter_channels, -1)

#         x4 = torch.matmul(x3, x2).view(batch_size, inter_channels, 1, 1)
#         y = self.conv4(x4) * x1
        
#         return x + y



# Test function
def test_context_aggregation():
    # Parameters
    in_channels = 8  # Example number of input channels
    reduction = 2    # Example reduction factor

    # Initialize module
    context_aggregation = ContextAggregation(in_channels, reduction=reduction)

    # Create x1 sample input tensor (batch_size, channels, height, width)
    input_tensor = torch.rand(2, in_channels, 16, 16)  # Example input

    # Forward pass
    output_tensor = context_aggregation(input_tensor)

    # Check the output shape
    assert output_tensor.shape == input_tensor.shape, "Output tensor shape does not match input shape."

    # Print output shape
    print(f"Output tensor shape: {output_tensor.shape}")

# Run the test function
if __name__ == "__main__":
    test_context_aggregation()