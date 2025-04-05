# ALSS-YOLO: An Adaptive Lightweight Channel Split and Shuffling Network for TIR Wildlife Detection in UAV Imagery
📝[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10680397)
## 引用

```
@article{he2024alss,
  title={ALSS-YOLO: An Adaptive Lightweight Channel Split and Shuffling Network for TIR Wildlife Detection in UAV Imagery},
  author={He, Ang and Li, Xiaobo and Wu, Ximei and Su, Chengyue and Chen, Jing and Xu, Sheng and Guo, Xiaobin},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2024},
  publisher={IEEE}
}
```
[模型整体配置](../ultralytics/cfg_yaml/models/ALSS-YOLO/ALSSn.yaml)

## 主要创新点汇总


### ALSS模块
```python
class ALSS(nn.Module):
    def __init__(self, C_in, C_out, num_blocks=1, alpha=0.2, beta=1, stride=1, use_identity=False, shortcut_mode=False):
        super(ALSS, self).__init__()
        
        # Calculate split sizes
        self.shortcut_channels = int(C_in * alpha)
        self.main_in_channels = C_in - self.shortcut_channels
        bottleneck_channels = int(self.main_in_channels * beta)
        main_out_channels = C_out - self.shortcut_channels
        
        self.num_blocks = num_blocks
        
        # Shortcut path
        if stride == 2:
            if shortcut_mode == 0:
                self.shortcut = Conv(self.shortcut_channels, self.shortcut_channels, 3, 2)
            elif shortcut_mode == 1:
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                    Conv(self.shortcut_channels, self.shortcut_channels, 3, 1)
                )
            else:
                self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.shortcut = nn.Identity() if use_identity else \
                Conv(self.shortcut_channels, self.shortcut_channels, 3, 1)

        
        # Main path
        self.initial_conv = Conv(self.main_in_channels, bottleneck_channels, 3, 1)
        
        self.middle_convs = nn.ModuleList()
        if stride == 2:
            self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 2, g=bottleneck_channels, act=False))
            for _ in range(1, num_blocks):
                self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 1, g=bottleneck_channels, act=False))
        else:
            for _ in range(num_blocks):
                self.middle_convs.append(Conv(bottleneck_channels, bottleneck_channels, 3, 1, g=bottleneck_channels, act=False))
        
        self.final_conv = Conv(bottleneck_channels, main_out_channels, 3, 1)

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
```
### LCA模块
```python
class LCA(nn.Module):
    def __init__(self, input_channel, reduction=32):
        super(LCA, self).__init__()
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
```
### 单通道focus
针对单通道图片整体前处理进行大的改动支持手动传参输入通道 ch: 1
```python
class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))
```

### 损失
代码参考：
```python
          elif FineSIoU:

                center_dist = torch.pow(center_width ** 2 + center_height ** 2, 0.5) #中心点距离


                """ #角度成本 """
                sin_w = torch.abs(center_width) / center_dist #中心点距离宽比中心点距离
                sin_h = torch.abs(center_height) / center_dist #中心点距离高比中心点距离
                threshold = 2 ** 0.5 / 2    #阈值
                sin_best = torch.where(sin_w > threshold, sin_h, sin_w) #根据角度选合适的

                angle_param = torch.cos(torch.arcsin(sin_best) * 2 - math.pi / 2)  #角度成本 0 -》0  45 ->1
                angle_cost = torch.pow(1 - torch.exp(-(angle_param-0.1736)),3)  #5度 角度成本系数


                """ 距离成本  原始  中心差比包围"""
                rho_x1 = (center_width / convex_width) ** 2
                rho_y1 = (center_height / convex_height) ** 2
                gamma =  angle_param -2
                distance_cost = 2 - torch.exp(-gamma * rho_x1) - torch.exp(-gamma * rho_y1)


                """ 形状成本    差值比最大"""
                omiga_w = torch.abs(w1 - w2) / w1 #两者宽差比宽最大的
                omiga_h = torch.abs(h1 - h2) / h1
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 6) + torch.pow(1 - torch.exp(-1 * omiga_h), 6) #形状成本系数
                if Focal:
                    return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, pow), torch.pow(inter/(union + eps), gamma) # Focal_SIou
                else:
                    return iou - torch.pow(0.5 * (angle_cost + distance_cost + shape_cost) + eps, pow)
```
