import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward
# 按照这个第三方库需要安装pip install pytorch_wavelets==1.3.0
# 如果提示缺少pywt库则安装 pip install PyWavelets
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







# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    

    
    
    
    input = torch.rand(1, 32, 64, 64) 

    # model = FFCM(dim=32)
    # model = HWD(in_ch=32, out_ch=64)  # 输入通道数，输出通道数
    # model = CED(32)
    # model = LRCED(32)
    model = InceptionDWConv2d(32)
    model = InceptionNeXtBlock(32)

    
    
    output = model(input) #torch.Size([1, 32, 64, 64])
    print('input :',input.size())
    print('output :', output.size())


