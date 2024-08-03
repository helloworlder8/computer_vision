# from random import weibullvariate
# import torch
# import torch.nn as nn
 
# class AFF(nn.Module):
# 	"""
# 	Implimenting AFF module
# 	"""
 
# 	def __init__(self, channels=64, r=4):
# 		super(AFF, self).__init__()
# 		inter_channels = int(channels //r )
 
# 		self.local_att = nn.Sequential(
# 			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(inter_channels),
# 		#	nn.ReLU(inplace=True),
#                         nn.SiLU(),
# 			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(channels),
# 		)
 
# 		self.global_att = nn.Sequential(
# 			nn.AdaptiveAvgPool2d(1),
# 			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(inter_channels),
# 		#	nn.ReLU(inplace=True),
#                         nn.SiLU(),
# 			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(channels),
# 		)
 
# 		self.sigmoid = nn.Sigmoid()
 
# 	def forward(self, input):
# 		x= input[0]
# 		y= input[1]
# 		xa = x + y
# 		xl = self.local_att(xa)
# 		xg= self.global_att(xa)
# 		xlg = xl + xg
# 		m = self.sigmoid(xlg)
 
# 		x_union_y = 2* x * m + 2* y * (1-m)
 
# 		return x_union_y
 
 
 
# class iAFF(nn.Module):
 
# 	"""
# 	implimenting iAFF module
# 	"""
 
# 	def __init__(self, channels=64, r=4):
# 		super(iAFF, self).__init__()
# 		inter_channels = int(channels // r)
 
# 		self.local_attention1 = nn.Sequential(
# 			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(inter_channels),
# 		#	nn.ReLU(inplace=True),
#                         nn.SiLU(),
# 			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(channels),
# 		)
# 		self.global_attention1 = nn.Sequential(
# 			nn.AdaptiveAvgPool2d(1),
# 			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(inter_channels),
# 		#	nn.ReLU(inplace=True),
#                         nn.SiLU(),
# 			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(channels),
# 		)
 
# 		self.local_attention2 = nn.Sequential(
# 			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(inter_channels),
# 		#	nn.ReLU(inplace=True),
#                         nn.SiLU(),
# 			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(channels),
# 		)
# 		self.global_attention2 = nn.Sequential(
# 			nn.AdaptiveAvgPool2d(1),
# 			nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(inter_channels),
# 		#	nn.ReLU(inplace=True),
#                         nn.SiLU(),
# 			nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
# 			nn.BatchNorm2d(channels),
# 		)
 
# 		self.sigmoid = nn.Sigmoid()
 
 
# 	def forward(self, x, y):
# 		"""
# 		Implimenting the iAFF forward step
# 		"""
 

# 		xa = x+y #torch.Size([8, 64, 32, 32])  torch.Size([8, 64, 32, 32])
# 		xl = self.local_attention1(xa) #torch.Size([8, 64, 32, 32])
# 		xg = self.global_attention1(xa) #torch.Size([8, 64, 1, 1])
# 		xlg = xl+xg #torch.Size([8, 64, 32, 32])
# 		m1 = self.sigmoid(xlg)
# 		xuniony = x * m1 + y * (1-m1) # torch.Size([8, 64, 32, 32])
 
# 		xl2 = self.local_attention2(xuniony)
# 		xg2 = self.global_attention2(xuniony)
# 		xlg2 = xl2 + xg2
# 		m2 = self.sigmoid(xlg2)
# 		z = x * m2 + y * (1-m2)
# 		return z
 
 
# if __name__ == '__main__':
# 	import os
# 	x = torch.randn(8,64,32,32)
# 	y = torch.randn(8,64,32,32)
# 	channels = x.shape[1]
 
# 	model = iAFF(channels=channels)
# 	output = model(x,y)
# 	print(output)
# 	print(output.shape)

import math

import numpy as np
import torch
import torch.nn as nn
from ultralytics.nn.modules import Conv
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


class CoordAtt(nn.Module):
    def __init__(self, input_channel, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        inter_channel = max(8, input_channel // reduction)
 
        self.conv1 = nn.Conv2d(input_channel, inter_channel, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inter_channel)
        self.act = h_swish()
 
        self.conv_h = nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1, padding=0,groups=input_channel)
        self.conv_w = nn.Conv2d(inter_channel, input_channel, kernel_size=1, stride=1, padding=0,groups=input_channel)


        
	def forward(self, x):  # torch.Size([2, 32, 64, 64])
		identity = x

		b, c, h, w = x.size()
		x_h = self.pool_h(x)  # torch.Size([2, 32, 64, 1])
		x_w = self.pool_w(x).permute(0, 1, 3, 2)  # torch.Size([2, 32, 1, 64]) -> torch.Size([2, 32, 64, 1])

		y = torch.cat([x_h, x_w], dim=2)  # torch.Size([2, 32, 128, 1])
		y = feature_shuffle(y, 4)  # torch.Size([2, 32, 128, 1])
		x_h, x_w = torch.split(y, [h, w], dim=2)  # ([2, 32, 64, 1])
		x_w = x_w.permute(0, 1, 3, 2)

		a_h = self.conv_h(x_h).sigmoid()  # 添加激活函数sigmoid
		a_w = self.conv_w(x_w).sigmoid()  # 使用self.conv_w并添加激活函数sigmoid

		out = identity * a_w * a_h  # torch.Size([2, 32, 64, 64])

		return out

######################  CoordAtt  ####     end   by  AI&CV  ###############################



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