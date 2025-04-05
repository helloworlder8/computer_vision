import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
# 按照这个第三方库需要安装pip install pytorch_wavelets==1.3.0
# 如果提示缺少pywt库则安装 pip install PyWavelets
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from mmcv.cnn import build_norm_layer
act_layer = nn.ReLU
ls_init_value = 1e-6



# 定义自定义的 __repr__ 方法，不修改注释
original_repr = torch.Tensor.__repr__
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
torch.Tensor.__repr__ = custom_repr



class FourierUnit(nn.Module):
    """
    输入：形状为 [batch, c, h, w] 的张量。
    流程：
    对输入张量进行二维实数傅里叶变换，得到频域表示。
    在频域中分解实部和虚部，并进行卷积等操作。
    将处理后的频域结果通过逆傅里叶变换还原到时域。
    输出：形状为 [batch, c, h, w] 的张量。
    """
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv = nn.Conv2d(
            in_channels=in_channels * 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=self.groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): #torch.Size([1, 64, 64, 64])
        b, c, h, w = x.size()
        """ 傅里叶变化 """
        x_fft = torch.fft.rfft2(x, norm='ortho') #->torch.Size([1, 64, 64, 33])
        
        real_part = torch.unsqueeze(torch.real(x_fft), dim=-1) #->torch.Size([1, 64, 64, 33, 1])
        imag_part = torch.unsqueeze(torch.imag(x_fft), dim=-1) #->torch.Size([1, 64, 64, 33, 1])
        fft_features = torch.cat((real_part, imag_part), dim=-1) #->torch.Size([1, 64, 64, 33, 2])
        fft_features = fft_features.permute(0, 1, 4, 2, 3).contiguous().view(b, -1, h, w//2 + 1) #->torch.Size([1, 128, 64, 33])
        """ 卷积 """
        fft_features = self.conv(fft_features) #->torch.Size([1, 128, 64, 33])
        
        fft_features = self.relu(self.bn(fft_features)) #->torch.Size([1, 128, 64, 33])
        fft_features = fft_features.view(b, -1, 2, h, w//2 + 1)
        fft_features = fft_features.permute(0, 1, 3, 4, 2).contiguous() #->torch.Size([1, 64, 64, 33, 2])
        fft_complex = torch.view_as_complex(fft_features) #->torch.Size([1, 64, 64, 33])
        """ 傅里叶逆变化 """
        output = torch.fft.irfft2(fft_complex, s=(h, w), norm='ortho') #->torch.Size([1, 64, 64, 64])
        return output


class Freq_Fusion(nn.Module):
    """
    输入：空间域张量
    流程：通过两路卷积提取特征后拼接，再通过FourierUnit模块获取频域特征并与原始特征相加，最后归一化和激活。
    输出：融合后的特征张量
    """
    def __init__(self, dim, kernel_size=[1,3,5,7], se_ratio=4, local_size=8, scale_ratio=2, spilt_num=4):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.conv_init_1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.FFC = FourierUnit(self.dim*2, self.dim*2)
        self.bn = nn.BatchNorm2d(dim*2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x): #torch.Size([1, 64, 64, 64])
        x_1, x_2 = torch.split(x, self.dim, dim=1) #[torch.Size([1, 32, 64, 64]),torch.Size([1, 32, 64, 64])]
        x_1 = self.conv_init_1(x_1) #->torch.Size([1, 32, 64, 64])
        x_2 = self.conv_init_2(x_2) #->torch.Size([1, 32, 64, 64])
        x0 = torch.cat([x_1, x_2], dim=1) #->torch.Size([1, 64, 64, 64])
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))#->torch.Size([1, 64, 64, 64])
        return x

# (ECCV2024)FFCM傅里叶卷积混合模块
class FFCM(nn.Module):
    """
    输入：空间域特征张量
    流程：
    1. 利用点卷积提高通道数并分成两路深度可分离卷积提取局部特征。
    2. 利用Freq_Fusion模块在频域中融合全局特征。
    3. 通道注意力加权输出。
    输出：融合全局与局部特征的张量
    """
    def __init__(self, dim, token_mixer_for_gloal=Freq_Fusion, mixer_kernel_size=[1,3,5,7], local_size=8):
        super(FFCM, self).__init__()
        self.dim = dim
        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size, se_ratio=8, local_size=local_size)
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.dw_conv_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )

    def forward(self, x):
        x = self.conv_init(x) #->torch.Size([1, 64, 64, 64])
        x = list(torch.split(x, self.dim, dim=1)) #[torch.Size([1, 32, 64, 64]),torch.Size([1, 32, 64, 64])]
        x_local_1 = self.dw_conv_1(x[0]) #->torch.Size([1, 32, 64, 64])
        x_local_2 = self.dw_conv_2(x[0]) #->torch.Size([1, 32, 64, 64])
        x_gloal = self.mixer_gloal(torch.cat([x_local_1, x_local_2], dim=1)) #->torch.Size([1, 64, 64, 64])
        x = self.ca_conv(x_gloal) #->torch.Size([1, 32, 64, 64])
        x = self.ca(x) * x #torch.Size([1, 32, 64, 64])
        return x
# Sequential(
#   (0): AdaptiveAvgPool2d(output_size=1)
#   (1): Conv2d(32, 8, kernel_size=(1, 1), stride=(1, 1))
#   (2): GELU(approximate='none')
#   (3): Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))
#   (4): Sigmoid()
# )



class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        yL, yH = self.wt(x) ##torch.Size([1, 32, 64, 64])->torch.Size([1, 32, 32, 32]) [torch.Size([1, 32, 3, 32, 32])]
        y_HL = yH[0][:, :, 0, ::] #->torch.Size([1, 32, 32, 32])
        y_LH = yH[0][:, :, 1, ::] #->torch.Size([1, 32, 32, 32])
        y_HH = yH[0][:, :, 2, ::] #->torch.Size([1, 32, 32, 32])
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1) #->torch.Size([1, 128, 32, 32])
        x = self.conv_bn_relu(x) #->torch.Size([1, 64, 32, 32])
        return x


class CED(nn.Module):

    def __init__(self, dim, drop_path=0., norm_cfg=dict(type='BN') , **kwargs):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = build_norm_layer(norm_cfg, dim)[1]
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x) #->torch.Size([1, 32, 64, 64])
        x = self.norm(x)    # input (N, C, *)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C) ->torch.Size([1, 64, 64, 32])
        x = self.pwconv1(x) #全连接 先上后下
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LRCED(nn.Module):

    def __init__(self, dim, drop_path=0., dilation=3, norm_cfg= dict(type='BN') , **kwargs):
        super().__init__()

        self.dwconv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, dilation=1, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer())

        self.dwconv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=dim),
            build_norm_layer(norm_cfg, dim)[1],
            act_layer())

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = act_layer()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            ls_init_value * torch.ones((dim)), requires_grad=True) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv1(x) + x
        x = self.dwconv2(x) + x

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    



'''
来自CVPR 2024 顶会
即插即用模块： InceptionDWConv2d（IDC）和InceptionNeXtBlock （INB）

InceptionNeXt 模块的提出是为了解决在现代卷积神经网络（CNN）中使用大核卷积时，效率与性能之间的矛盾问题。
受到 Vision Transformer (ViT) 长距离建模能力的启发，许多研究表明大核卷积可以扩大感受野并提高模型性能。
例如，ConvNeXt 使用 7×7 深度卷积来提升效果。然而，尽管大核卷积在理论上 FLOPs 较少，
但其高内存访问成本导致在高性能计算设备上效率下降。并通过实验表明大核卷积在计算效率方面存在问题。

InceptionNeXt 模块通过引入高效的大核卷积分解技术，解决了大核卷积在计算效率上的问题。
其主要作用包括：
1.卷积分解：将大核卷积分解为多个并行的分支，其中包含小核卷积、带状核（1x11 和 11x1）以及身份映射，
使模型能够高效地利用大感受野，同时减少计算开销。
2.提高计算效率：通过分解卷积核来提升计算效率，减少大核卷积带来的高计算成本，实现速度与性能的平衡。
3.扩大感受野：带状核能够在保持较低计算成本的情况下扩大感受野，从而捕捉更多的空间信息。
4.性能优势：在不牺牲模型性能的前提下，InceptionNeXt 模块提高了推理速度，尤其适合高性能与高效率需求的场景。
InceptionNeXt 模块通过分解大核卷积的创新设计，在保持模型准确率的同时，显著提升了推理速度。

'''
class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x): #torch.Size([1, 32, 64, 64])
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )



class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class InceptionNeXtBlock(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer=InceptionDWConv2d,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x) #->torch.Size([1, 32, 64, 64])
        x = self.norm(x)
        x = self.mlp(x) #->torch.Size([1, 32, 64, 64]) 先上后下
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1)) #通道位置xuexi
        x = self.drop_path(x) + shortcut
        return x




def global_median_pooling(x):  
    # 全局中值池化操作
    median_pooled, _ = torch.median(x.view(x.size(0), x.size(1), -1), dim=2) #torch.Size([1, 8, 4096]) ->torch.Size([1, 8])
    median_pooled = median_pooled.view(x.size(0), x.size(1), 1, 1)
    return median_pooled 


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        # 平均池化和最大池化操作，用于生成不同的特征表示
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): #torch.Size([1, 8, 64, 64])
        # 平均池化
        avg_out = self.avg_pool(x) #torch.Size([1, 8, 1, 1])
        # 最大池化
        max_out = self.max_pool(x) #torch.Size([1, 8, 1, 1])
        # 均值池化
        med_out = global_median_pooling(x) #torch.Size([1, 8, 1, 1])
        # 将两个池化特征相加
        out = avg_out + max_out + med_out #torch.Size([1, 8, 1, 1])
        # 使用1x1卷积生成通道注意力图
        out = self.fc(out) #torch.Size([1, 8, 1, 1])
        # 生成通道注意力权重 
        x_ca = x*self.sigmoid(out)
        return x_ca #torch.Size([1, 8, 64, 64])


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        # 使用7x7卷积实现空间注意力
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): #torch.Size([1, 32, 64, 64])
        # 平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True) #torch.Size([1, 1, 64, 64])
        # 最大池化
        max_out, _ = torch.max(x, dim=1, keepdim=True) #torch.Size([1, 1, 64, 64])
        # 将二者拼接
        y = torch.cat([avg_out, max_out], dim=1) #torch.Size([1, 2, 64, 64])
        # 使用7x7卷积生成空间注意力图
        y = self.conv(y) #torch.Size([1, 1, 64, 64])
        # 生成空间注意力权重
        x_sa = x*self.sigmoid(y)
        return x_sa  


class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels, factor=0.25):
        super(MSAA, self).__init__()
        # 通道压缩
        dim = int(out_channels*factor)
        self.down = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)
        # 不同大小卷积核1x1,3x3,5x5和7x7提取多尺度特征
        self.conv_1x1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        # 引入空间和通道注意力模块
        self.spatial_attention = SpatialAttentionModule(kernel_size=7)
        self.channel_attention = ChannelAttentionModule(dim)
        # 通道升维
        self.up = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x): #torch.Size([1, 32, 64, 64])
        x = self.down(x) #torch.Size([1, 8, 64, 64])
        # 残差通道注意力机制CA
        x = x + self.channel_attention(x)
        # 提取多尺度卷积特征
        x_1x1 = self.conv_1x1(x) #torch.Size([1, 8, 64, 64])
        x_3x3 = self.conv_3x3(x) #torch.Size([1, 8, 64, 64])
        x_5x5 = self.conv_5x5(x) #torch.Size([1, 8, 64, 64])
        x_7x7 = self.conv_7x7(x) #torch.Size([1, 8, 64, 64])
        x_ms = torch.cat([x_1x1,x_3x3,x_5x5,x_7x7], dim=1) #torch.Size([1, 32, 64, 64])
        # 残差空间注意力机制SA
        x_s = x_ms + self.spatial_attention(x_ms) #torch.Size([1, 32, 64, 64])

        # 将处理后的特征与原始特征相加，并恢复到原始的通道数
        x_out = self.up(x_s) #->torch.Size([1, 32, 64, 64])
        return x_out  



class SCSA(nn.Module):
    def __init__(self, dim, head_num, window_size=7, group_kernel_sizes=[3, 5, 7, 9], qkv_bias=False,
                 fuse_bn=False, down_sample_mode='avg_pool', attn_drop_ratio=0, gate_layer='sigmoid'):
        super(SCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode
        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x): #torch.Size([1, 32, 64, 64])
        b, c, h_, w_ = x.size()
        # (B,C,H) 平均池化
        x_h = x.mean(dim=3) #torch.Size([1, 32, 64])
        # 特征拆分
        # H方向上的局部特征l_x_h和三个不同尺度的全局特征g_x_h_s, g_x_h_m, g_x_h_l
        # [1,8,256]
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1) #torch.Size([1, 8, 64]) torch.Size([1, 8, 64]) torch.Size([1, 8, 64])
        # (B,C,W)
        x_w = x.mean(dim=2)  #torch.Size([1, 32, 64])
        # W方向上的局部特征l_x_w和三个不同尺度的全局特征g_x_w_s, g_x_w_m, g_x_w_l
        # [1,8,256]
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1) #torch.Size([1, 8, 64]) torch.Size([1, 8, 64]) torch.Size([1, 8, 64])

        # 特征提取，空间注意力计算
        x_h_attn = self.sa_gate(self.norm_h(torch.cat((
            self.local_dwc(l_x_h), #torch.Size([1, 8, 64])
            self.global_dwc_s(g_x_h_s), #torch.Size([1, 8, 64])
            self.global_dwc_m(g_x_h_m), #torch.Size([1, 8, 64])
            self.global_dwc_l(g_x_h_l), #torch.Size([1, 8, 64])
        ), dim=1))) #->torch.Size([1, 32, 64])
        x_h_attn = x_h_attn.view(b, c, h_, 1) #->torch.Size([1, 32, 64, 1])

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_) #->torch.Size([1, 32, 1, 64])


        x = x * x_h_attn * x_w_attn #->torch.Size([1, 32, 64, 64])

        # 下采样，得到较小特征图
        y = self.down_func(x) #->torch.Size([1, 32, 9, 9])
        y = self.conv_d(y) #->torch.Size([1, 32, 9, 9])
        # [1,32,36,36]
        _, _, h_, w_ = y.size()

        # 特征归一化与变换
        y = self.norm(y) #组归一化 ->torch.Size([1, 32, 9, 9])
        # 查询q
        q = self.q(y) #->torch.Size([1, 32, 9, 9])
        # 键k
        k = self.k(y) #->torch.Size([1, 32, 9, 9])
        # 值v
        v = self.v(y) #->torch.Size([1, 32, 9, 9])

        # 注意力矩阵计算
        # 将查询和键进行点积操作，计算注意力矩阵，并应用缩放因子来防止数值溢出。
        # (B,C,H_,W_) -> (B, head_num, head_dim, N)   
        # [1,32,36,36]-->[1,8,4,36x36]
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim)) #通道变成多少个头以及头的维度  宽高统一  torch.Size([1, 8, 4, 81])
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim)) #torch.Size([1, 8, 4, 81])
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim)) #torch.Size([1, 8, 4, 81])
        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler #torch.Size([1, 8, 4, 4])
        # 通过门控机制操作计算得到注意力权重，并使用Dropout进行正则化处理。
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v #->torch.Size([1, 8, 4, 81])
        # (B, head_num, head_dim, N)-->(B,C,H_,W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B,C,1,1)
        attn = attn.mean((2, 3), keepdim=True) #torch.Size([1, 32, 1, 1])
        attn = self.ca_gate(attn)
        return attn * x #torch.Size([1, 32, 64, 64])





class SPRModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SPRModule, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)

        self.fc1 = nn.Conv2d(channels*5, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): #torch.Size([1, 8, 64, 64])
        out1 = self.avg_pool1(x).view(x.size(0), -1, 1, 1) #torch.Size([1, 8, 1, 1])
        out2 = self.avg_pool2(x).view(x.size(0), -1, 1, 1) #torch.Size([1, 8, 2, 2]) ->torch.Size([1, 32, 1, 1]) 
        out = torch.cat((out1, out2), 1) #->torch.Size([1, 40, 1, 1])
        out = self.fc1(out) #->torch.Size([1, 2, 1, 1])
        out = self.relu(out)
        out = self.fc2(out) #->torch.Size([1, 8, 1, 1])
        weight = self.sigmoid(out)
        return weight #->torch.Size([1, 8, 1, 1])



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MSAModule(nn.Module):
    def __init__(self, inplanes, scale=4, stride=1,):
        super(MSAModule, self).__init__()
        # 切成几块
        self.scale = scale
        # 每一块的通道数
        self.width = int(inplanes/scale)
        self.stride = stride

        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])

        for i in range(self.scale):
            self.convs.append(conv3x3(self.width, self.width, stride))
            self.bns.append(nn.BatchNorm2d(self.width))

        self.attention = SPRModule(self.width)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0] #1

        # HPC模块
        # torch.split()作用将tensor分成块结构。
        spx = torch.split(x, self.width, 1) #[torch.Size([1, 8, 64, 64]),torch.Size([1, 8, 64, 64]),torch.Size([1, 8, 64, 64]),c]
        # 对于每一块来说
        for i in range(self.scale):
            if i == 0:
                sp = spx[i] #torch.Size([1, 8, 64, 64])
            else:
                sp = sp + spx[i] #torch.Size([1, 8, 64, 64])
            sp = self.convs[i](sp) #torch.Size([1, 8, 64, 64])
            sp = self.bns[i](sp) #torch.Size([1, 8, 64, 64])

            if i == 0:
                out = sp #torch.Size([1, 8, 64, 64])
            else:
                out = torch.cat((out, sp), 1)
        feats = out #torch.Size([1, 32, 64, 64])
        feats = feats.view(batch_size, self.scale, self.width, feats.shape[2], feats.shape[3]) #torch.Size([1, 4, 8, 64, 64])

        # 对于每一块，执行SPR操作
        sp_inp = torch.split(out, self.width, 1)
        attn_weight = []
        for inp in sp_inp:
            attn_weight.append(self.attention(inp)) #torch.Size([1, 8, 64, 64])
        attn_weight = torch.cat(attn_weight, dim=1) #torch.Size([1, 32, 1, 1])
        attn_vectors = attn_weight.view(batch_size, self.scale, self.width, 1, 1) #torch.Size([1, 4, 8, 1, 1])

        # softmax操作
        attn_vectors = self.softmax(attn_vectors) #torch.Size([1, 4, 8, 1, 1])
        feats_weight = feats * attn_vectors #torch.Size([1, 4, 8, 64, 64])
 
        for i in range(self.scale):
            x_attn_weight = feats_weight[:, i, :, :, :]
            if i == 0:
                out = x_attn_weight
            else:
                out = torch.cat((out, x_attn_weight), 1)
        return out






def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    stage = nn.Sequential()
    pad = (ksize-1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, 
                                       kernel_size=ksize, 
                                       stride=stride,
                                       padding=pad, 
                                       bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ESELayer(nn.Module):
    """ Effective Squeeze-Excitation
    """
    def __init__(self, channels, act='hardsigmoid'):
        super(ESELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Sigmoid()

    def forward(self, x): #torch.Size([1, 24, 20, 20])
        x_se = x.mean((2, 3), keepdim=True) #torch.Size([1, 24, 1, 1])
        x_se = self.fc(x_se) #torch.Size([1, 24, 1, 1])
        return x * self.act(x_se)  #torch.Size([1, 24, 20, 20])


class Efficicent_ASFF(nn.Module):
    def __init__(self, level, rfb=True, vis=False):
        super(Efficicent_ASFF, self).__init__()
        self.level = level
        self.dim = [32, 32, 128]
        self.compress_c = 8 if rfb else 32
        self.init_layers()
        self.eca = ESELayer(3*self.compress_c)
        self.weight_levels = nn.Conv2d(self.compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def init_layers(self):
        if self.level == 0:
            self.stride_level_0 = add_conv(32, self.compress_c, 3, 1)
            self.stride_level_1 = add_conv(32, self.compress_c, 3, 2)
            self.stride_level_2 = add_conv(32, self.compress_c, 3, 2)
            self.expand = add_conv(self.compress_c, 32, 3, 1)
        elif self.level == 1:
            self.compress_level_0 = add_conv(32, self.compress_c, 1, 1)
            self.stride_level_1 = add_conv(32, self.compress_c, 3, 1)
            self.stride_level_2 = add_conv(128, self.compress_c, 3, 2)
            self.expand = add_conv(self.compress_c, 32, 3, 1)
        elif self.level == 2:
            self.compress_level_0 = add_conv(32, self.compress_c, 1, 1)
            self.compress_level_1 = add_conv(32, self.compress_c, 1, 1)
            self.stride_level_2 = add_conv(128, self.compress_c, 3, 1)
            self.expand = add_conv(self.compress_c, 128, 3, 1)

    def resize_features(self, x_level_0, x_level_1, x_level_2): #torch.Size([1, 32, 20, 20]) torch.Size([1, 32, 40, 40]) torch.Size([1, 32, 80, 80])
        if self.level == 0:
            level_0_resized = self.stride_level_0(x_level_0) #-》torch.Size([1, 8, 20, 20])
            level_1_resized = self.stride_level_1(x_level_1) #-》torch.Size([1, 8, 20, 20])
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1) #->torch.Size([1, 32, 40, 40])
            level_2_resized = self.stride_level_2(level_2_downsampled_inter) #-》torch.Size([1, 8, 20, 20])
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = self.stride_level_2(x_level_2)
        return level_0_resized, level_1_resized, level_2_resized

    def forward(self, x_level_0, x_level_1, x_level_2): #torch.Size([1, 32, 20, 20]) torch.Size([1, 32, 40, 40]) torch.Size([1, 32, 80, 80])
        # 维度压缩
        resized_features = self.resize_features(x_level_0, x_level_1, x_level_2)
        level_0_resized, level_1_resized, level_2_resized = resized_features

        # 高效的残差通道注意力机制ERCA
        levels_features = torch.cat((level_0_resized, level_1_resized, level_2_resized), 1) #torch.Size([1, 24, 20, 20])
        levels_features_erca = levels_features + self.eca(levels_features)

        # 高效的空间注意力机制ESA
        levels_weight = self.weight_levels(levels_features_erca) #->torch.Size([1, 3, 20, 20])
        levels_weight = F.softmax(levels_weight, dim=1) #->torch.Size([1, 3, 20, 20])
        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:] + \
                            level_1_resized * levels_weight[:,1:2,:,:] + \
                            level_2_resized * levels_weight[:,2:,:,:] #->torch.Size([1, 8, 20, 20])
        # 维度扩张
        out = self.expand(fused_out_reduced)
        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class Lite_ASFF(nn.Module):
    def __init__(self, level, rfb=True, vis=False):
        super(Lite_ASFF, self).__init__()
        self.level = level
        self.dim = [32, 32, 128]
        self.inter_dim = self.dim[self.level]
        self.init_layers()
        compress_c = 8 if rfb else 16
        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def init_layers(self):
        if self.level == 0:
            self.stride_level_1 = add_conv(32, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 32, 3, 1)
        elif self.level == 1:
            self.compress_level_0 = add_conv(32, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 32, 3, 1)
        elif self.level == 2:
            self.compress_level_0 = add_conv(32, self.inter_dim, 1, 1)
            self.compress_level_1 = add_conv(32, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 128, 3, 1)

    def resize_features(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
        return level_0_resized, level_1_resized, level_2_resized

    def forward(self, x_level_0, x_level_1, x_level_2):
        resized_features = self.resize_features(x_level_0, x_level_1, x_level_2)
        level_0_resized, level_1_resized, level_2_resized = resized_features
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:] + \
                            level_1_resized * levels_weight[:,1:2,:,:] + \
                            level_2_resized * levels_weight[:,2:,:,:]
        out = self.expand(fused_out_reduced)
        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out



class ASFF(nn.Module):
    def __init__(self, level, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        # 特征金字塔从上到下三层的channel数
        # 对应特征图大小(以640*640输入为例)分别为20*20, 40*40, 80*80
        self.dim = [512, 32, 128]
        self.inter_dim = self.dim[self.level]
        # 特征图大小最小的一层，channel数512
        if level==0: 
            self.stride_level_1 = add_conv(32, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        # 特征图大小适中的一层，channel数256
        elif level==1: 
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(128, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 32, 3, 1)
        # 特征图大小最大的一层，channel数128
        elif level==2: 
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1)
            self.compress_level_1 = add_conv(32, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 128, 3, 1)

        compress_c = 8 if rfb else 16  

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
        



def autopad(k, p=None, d=1):  
    # kernel, padding, dilation
    # 填充Pad至相同same形状输出
    if d > 1:
        # actual kernel-size
        k = d * (k-1) + 1 if isinstance(k, int) else [d * (x-1) + 1 for x in k]  
    # auto-pad
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  
    return p

class Conv(nn.Module):
    # 标准卷积 Standard convolution
    default_act = nn.SiLU()  
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
class DWConv(Conv):
    # 深度卷积 Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)
        
class ESELayer(nn.Module):
    """ Effective Squeeze-Excitation
    """
    def __init__(self, channels, act='hardsigmoid'):
        super(ESELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

class LightConv(nn.Module):
    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """
    def __init__(self, c1, cm, c2, k=3, n=5, lightconv=False, shortcut=True, act=nn.ReLU()):
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        # 聚合aggregation conv
        self.agg_conv = Conv(c1 + n*cm, c2, 1, 1, act=act)  
        # ECA注意力
        self.eca = ESELayer(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x): #torch.Size([1, 16, 64, 64])
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.agg_conv(torch.cat(y, 1))
        y = self.eca(y)
        return y + x if self.add else y

class C2f_HGNetv1(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, extra=2, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  
        self.cv1 = Conv(c1, 2*self.c, 1, 1)
        self.cv2 = Conv((2+n) * self.c, c2, 1)  
        self.m = nn.Sequential(*(HGBlock(self.c, self.c, self.c) for _ in range(n)))

    def forward(self, x): #torch.Size([1, 32, 64, 64])
        y = list(self.cv1(x).chunk(2, 1)) #[torch.Size([1, 16, 64, 64]), torch.Size([1, 16, 64, 64])]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    

    
    
    
    input = torch.rand(1, 32, 64, 64) 

    # model = FFCM(dim=32)
    # model = HWD(in_ch=32, out_ch=64)  # 输入通道数，输出通道数
    # model = CED(32)
    # model = LRCED(32)
    model = InceptionDWConv2d(32)
    model = InceptionNeXtBlock(32)
    model = SCSA(dim=32, head_num=8, window_size=7)
    # model = MSAA(32, 32)
    model = MSAModule(inplanes=32)
    
    output = model(input) #torch.Size([1, 32, 64, 64])
    print('input :',input.size())
    print('output :', output.size())


