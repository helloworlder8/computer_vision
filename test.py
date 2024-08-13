import torch
import torch.nn as nn

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # Reshape
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # Transpose to shuffle channels
    x = x.transpose(1, 2).contiguous()
    # Flatten
    x = x.view(batch_size, -1, height, width)
    return x

class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, groups=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=kernel_size//2, groups=groups)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))
    
class ALSS(nn.Module):
    def __init__(self, C_in, C_out, alpha=0.2, beta=1, stride=1, use_identity=False, use_pool=False, num_blocks=1):
        super(ALSS, self).__init__()
        
        # Calculate split sizes
        self.shortcut_channels = int(C_in * alpha)
        self.main_in_channels = C_in - self.shortcut_channels
        bottleneck_channels = int(self.main_in_channels * beta)
        main_out_channels = C_out - self.shortcut_channels
        
        self.num_blocks = num_blocks
        
        # Shortcut path
        if stride == 1:
            self.shortcut = nn.Identity() if use_identity else Conv(self.shortcut_channels, self.shortcut_channels, 3, stride=1)
        else:
            if use_pool:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    Conv(self.shortcut_channels, self.shortcut_channels, 3, stride=1)
                )
            else:
                self.shortcut = Conv(self.shortcut_channels, self.shortcut_channels, 3, stride=2)
        
        # Main path
        self.initial_conv = Conv(self.main_in_channels, bottleneck_channels, 3, stride=1)
        
        self.middle_convs = nn.ModuleList()
        if stride == 2:
            self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, stride=2, groups=bottleneck_channels, act=False))
            for _ in range(1, num_blocks):
                self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, stride=1, groups=bottleneck_channels, act=False))
        else:
            for _ in range(num_blocks):
                self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, stride=1, groups=bottleneck_channels, act=False))
        
        self.final_conv = Conv(bottleneck_channels, main_out_channels, 3, stride=1)

    def forward(self, x):
        # Split input into shortcut and main branches
        proportional_sizes=[self.shortcut_channels,self.main_in_channels]
        x = list(x.split(proportional_sizes, dim=1))

        
        # Process shortcut path
        shortcut_x = self.shortcut(x[0])
        
        # Process main path
        main_x = self.initial_conv(x[1])
        for conv in self.middle_convs:
            main_x = conv(main_x)
        main_x = self.final_conv(main_x)
        
        # Concatenate and shuffle channels
        out_x = torch.cat((main_x, shortcut_x), dim=1)
        out_x = channel_shuffle(out_x, 2)
        return out_x





# 测试程序
def test_alss():
    # 输入参数设置
    C_in = 64
    C_out = 128
    input_size = (1, C_in, 32, 32)  # (batch_size, channels, height, width)
    x = torch.randn(input_size)
    
    # 创建模型实例
    model = ALSS(C_in=C_in, C_out=C_out, num_blocks=2, alpha=0.25, stride=1, use_identity=False, use_pool=False, beta=0.5)
    
    # 前向传播
    output = model(x)
    
    # 输出信息
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # 检查输出是否符合预期
    assert output.shape == (1, C_out, 32, 32), "Output shape is incorrect!"
    print("Test passed successfully.")

if __name__ == "__main__":
    test_alss()
    
    
