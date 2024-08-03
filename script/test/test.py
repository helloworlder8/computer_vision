# import torch
# import torch.nn as nn

# class SpaceToDepth(nn.Module):
#     # Changing the dimension of the Tensor
#     def __init__(self, dimension=1):
#         super().__init__()
#         self.d = dimension
 
#     def forward(self, x):
#         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], self.d) #torch.Size([1, 2, 4, 4]) torch.Size([1, 2, 2, 2])

# # Define an example input tensor of shape (batch_size, channels, height, width)
# # For this example, we'll use a tensor with a shape of (1, 2, 4, 4)
# input_tensor = torch.tensor([[
#     [[1, 2, 3, 4],
#      [5, 6, 7, 8],
#      [9, 10, 11, 12],
#      [13, 14, 15, 16]],
    
#     [[17, 18, 19, 20],
#      [21, 22, 23, 24],
#      [25, 26, 27, 28],
#      [29, 30, 31, 32]]
# ]])

# print("Input Tensor:\n", input_tensor)

# # Create an instance of the SpaceToDepth class
# space_to_depth = SpaceToDepth(dimension=1)

# # Apply the transformation
# output_tensor = space_to_depth(input_tensor)

# print("Output Tensor:\n", output_tensor.shape)



# import torch

# # 假设x是一个有10个通道的张量
# x = torch.randn(1, 10, 24, 24)  # 示例张量，大小为(1, 10, 24, 24)

# # 假设我们想按3:7的比例分割通道
# total_channels = x.size(1)
# ratio = [3, 7]
# proportional_sizes = [int(total_channels * (r / sum(ratio))) for r in ratio]

# # 按计算出的比例分割张量
# y = list(x.split(proportional_sizes, dim=1))

# print(f"Original tensor shape: {x.shape}")
# print(f"First chunk shape: {y[0].shape}")
# print(f"Second chunk shape: {y[1].shape}")





#1
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Conv(nn.Module):
#     # 假设你已有Conv类的实现
#     def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, k, stride=s, padding=p)
#         # 添加其他必要的层或操作，比如激活函数

#     def forward(self, x):
#         x = self.conv(x)
#         # 添加其他必要的操作
#         return x

# ###################### PConv  ####     start#############################
# class PConv(nn.Module):
#     def __init__(self, dim, ouc, n_div=4, forward='split_cat'): #64 32 4  'split_cat'
#         super().__init__()
#         self.dim_conv3 = dim // n_div #16
#         self.dim_untouched = dim - self.dim_conv3  #48
#         self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False) #16 16 3 1 1
#         self.conv = Conv(dim, ouc, k=1) #64 32 1
 
#         if forward == 'slicing':
#             self.forward = self.forward_slicing
#         elif forward == 'split_cat':
#             self.forward = self.forward_split_cat
#         else:
#             raise NotImplementedError
 
#     def forward_slicing(self, x): #没有本质差异
#         # only for inference
#         x = x.clone()   # !!! Keep the original input intact for the residual connection later
#         x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :]) #torch.Size([1, 16, 64, 64])
#         x = self.conv(x)
#         return x
 
#     def forward_split_cat(self, x):  #torch.Size([1, 64, 64, 64])
#         # for training/inference
#         x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1) #torch.Size([1, 16, 64, 64]) torch.Size([1, 48, 64, 64])
#         x1 = self.partial_conv3(x1) #torch.Size([1, 16, 64, 64])
#         x = torch.cat((x1, x2), 1)#torch.Size([1, 64, 64, 64])
#         x = self.conv(x)
#         return x
 
 
# ###################### PConv  ####     end#############################

# def test_pconv_slicing():
#     # 输入特征维度，输出通道数
#     dim, ouc = 64, 32

#     # 创建PConv实例，使用slicing作为前向传播策略
#     pconv = PConv(dim, ouc, forward='slicing')

#     # 创建随机输入数据
#     x = torch.randn(1, dim, 64, 64)  # 假设输入大小为 (1, 64, 64, 64)

#     # 通过PConv实例运行数据
#     output = pconv(x)

#     # 检查输出形状
#     print("Output shape:", output.shape)

# if __name__ == "__main__":
#     test_pconv_slicing()










# import torch

# # 假设x是一个有10个通道的张量
# x = torch.randn(1, 10, 24, 24)  # 示例张量，大小为(1, 10, 24, 24)

# # 假设我们想按3:7的比例分割通道
# total_channels = x.size(1)
# ratio = [3, 7]
# proportional_sizes = [int(total_channels * (r / sum(ratio))) for r in ratio]

# # 按计算出的比例分割张量
# y = list(x.split(proportional_sizes, dim=1))

# print(f"Original tensor shape: {x.shape}")
# print(f"First chunk shape: {y[0].shape}")
# print(f"Second chunk shape: {y[1].shape}")



def adjust_groups(in_channels, initial_groups):
    """
    调整分组数以确保输入通道数能被分组数整除。
    如果初始分组数不能整除输入通道数，找到最大的临近值作为新的分组数。

    参数:
    - in_channels (int): 输入通道数。
    - initial_groups (int): 初始的分组数。

    返回:
    - int: 调整后的分组数。
    """
    if in_channels % initial_groups == 0:
        return initial_groups  # 如果能整除，直接返回初始分组数
    else:
        # 找到能被in_channels整除的最大临近值
        for groups in range(initial_groups, 0, -1):
            if in_channels % groups == 0:
                return groups

# 示例使用
in_channels = 25  # 假设的输入通道数
initial_groups = 4  # 假设的初始分组数

# 调用函数以获取调整后的分组数
adjusted_groups = adjust_groups(in_channels, initial_groups)
print(f"Adjusted groups: {adjusted_groups}")

