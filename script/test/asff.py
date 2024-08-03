import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加卷积层的函数，这里简单定义为 nn.Conv2d
def add_conv(in_channels, out_channels, kernel_size, stride, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding)

class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False): #0 false false
        super(ASFF, self).__init__()
        self.level = level #0
        self.dim = [512, 256, 256]
        self.inter_dim = self.dim[self.level] #512
        if level == 0:
            self.upchannal_downsize = add_conv(256, self.inter_dim, 3, 2,1) #256 512  3 2 1
            self.channal_change = add_conv(self.inter_dim, 1024, 3, 1,1)        #512 1024 3 1


        elif level == 1:
            self.down_channal = add_conv(512, self.inter_dim, 1, 1)
            self.upchannal_downsize = add_conv(256, self.inter_dim, 3, 2,1)
            self.channal_change = add_conv(self.inter_dim, 512, 3, 1,1)



        elif level == 2:
            self.down_channal = add_conv(512, self.inter_dim, 1, 1)
            self.channal_change = add_conv(self.inter_dim, 256, 3, 1,1)
 

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory  16


        self.down_channal_toc = add_conv(self.inter_dim, compress_c, 1, 1) #512 16 1 1
        self.down_channal_to_3 = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0) #48 3 1 1 0
        self.vis = vis #false
 
    def forward(self, x_0, x_1, x_2):  # (b,512,13,13),(b,256,26,26),(b,256,52,52)
        if self.level == 0:
            x_0_resized = x_0                                                           #       (b,512,13,13)
            x_1_resized = self.upchannal_downsize(x_1)                                        # (b,512,13,13)
            x_2_resized = self.upchannal_downsize(F.max_pool2d(x_2, 3, stride=2, padding=1) )  #(b,512,13,13)

        elif self.level == 1:
            x_0_resized = F.interpolate(self.down_channal(x_0) , scale_factor=2, mode='nearest')  # (b,256,26,26)
            x_1_resized = x_1
            x_2_resized = self.upchannal_downsize(x_2)  # (b,256,26,26)


        elif self.level == 2:
            x_0_resized = F.interpolate(self.down_channal(x_0), scale_factor=4, mode='nearest')  # (b,256,52,52)
            x_1_resized = F.interpolate(x_1, scale_factor=2, mode='nearest')  # (b,256,52,52)
            x_2_resized = x_2
 
        x_0_weight = self.down_channal_toc(x_0_resized)  # (b,16,13,13) #维持          通道压缩
        x_1_weight = self.down_channal_toc(x_1_resized)  # (b,16,13,13) #加通道减尺寸   通道压缩
        x_2_weight = self.down_channal_toc(x_2_resized)  # (b,16,13,13) #
        x_contact_weight = torch.cat((x_0_weight, x_1_weight, x_2_weight), 1)  # (b,48,13,13) 通道拼接
        x_contact_weight = self.down_channal_to_3(x_contact_weight)  # (b,3,13,13) 通道压缩
        x_contact_weight = F.softmax(x_contact_weight, dim=1) #torch.Size([2, 3, 13, 13]) 生成通道权重
        #([2, 512, 13, 13]) torch.Size([2, 1, 13, 13])   ->torch.Size([2, 512, 13, 13])
       # 自适应权重融合
        fused = x_0_resized * x_contact_weight[:,0:1,:,:]+\
                            x_1_resized * x_contact_weight[:,1:2,:,:]+\
                            x_2_resized * x_contact_weight[:,2:,:,:] 

 
        level_y = self.channal_change(fused)  # (b,1024,13,13) 通道扩充
 
        if self.vis:
            return level_y, x_contact_weight, fused.sum(dim=1)
        else:
            return level_y

# 测试代码
def test_asff():
    # 定义输入数据
    x_0 = torch.randn(2, 512, 13, 13)#以第一个为标准
    x_1 = torch.randn(2, 256, 26, 26)
    x_2 = torch.randn(2, 256, 52, 52)

    # 创建 ASFF 模型实例
    asff = ASFF(level=2, rfb=False, vis=False)
    #torch.Size([2, 1024, 13, 13])  torch.Size([2, 512, 26, 26]) torch.Size([2, 256, 52, 52])
    # 运行模型
    output = asff(x_0, x_1, x_2)

    # 打印输出的形状
    print("Output shape:", output.shape)

# 运行测试
test_asff()
