from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    """Implements a residual bottleneck block with downsampling and expansion for deep neural networks."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        """Initializes the Bottleneck module with given input planes, output planes, and stride."""
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):
        """Process input tensor `x` through the defined network layers and return the output tensor."""
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    """Applies multi-head attention pooling over 2D spatial data, transforming it into a fixed-size output embedding."""

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        """Initializes AttentionPool2d with spatial dimension, embedding dimension, number of heads, and optional output
        dimension.
        """
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        """Executes the forward pass of the model using multi-head attention on input tensor 'x', returning the
        processed data.
        """
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:

    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        """Initialize model with customizable layers, output dimensions, attention heads, input resolution, and width
        parameters.
        """
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        """Constructs a sequential layer of Bottleneck blocks with the given planes, number of blocks, and stride."""
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        layers.extend(Bottleneck(self._inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the network stem, applying convolutions, batch normalization, ReLU activations, and
        average pooling.
        """

        def stem(x):
            """Forward pass through the network stem, applying convolutions, batch normalization, ReLU activations, and
            average pooling.
            """
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        """Performs forward pass through the LayerNorm, converting input to float32 and back to its original type."""
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """Applies the QuickGELU activation function, a faster approximation of GELU, to an input tensor."""

    def forward(self, x: torch.Tensor):
        """Applies the QuickGELU activation function to an input tensor."""
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Implements a residual attention block with multi-head attention and MLP layers for transformer models."""

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        """Initializes the ResidualAttentionBlock with model dimension, number of heads, and optional attention mask."""
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        """Compute scaled dot-product attention using query, key, and value tensors, with optional attention mask
        adjustment.
        """  #torch.Size([50, 1, 768])
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """Performs forward pass through the network, applying attention and MLP layers sequentially."""
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Processes input tensors through multiple residual attention blocks for sequence modeling tasks."""

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        """Initializes the Transformer model with specified width, layers, heads, and optional attention mask."""
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        """Process the input tensor 'x' through a sequence of residual attention blocks."""
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """Vision Transformer model for image classification using patch embeddings and multi-head self-attention."""

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        """Initialize a VisionTransformer with given input resolution, patch size, width, layers, heads, and output
        dimension.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor): #torch.Size([1, 3, 224, 224])
        """Processes input tensor through embedding, layer normalization, and transformer layers."""
        x = self.conv1(x)  # shape = [*, width, grid, grid] #等效打散后加一层cnn  torch.Size([1, 768, 7, 7])
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2] #torch.Size([1, 768, 49])
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width] torch.Size([1, 50, 768])
        x = x + self.positional_embedding.to(x.dtype) #torch.Size([1, 50, 768])
        x = self.ln_pre(x) #torch.Size([1, 50, 768])

        x = x.permute(1, 0, 2)  # torch.Size([50, 1, 768]) 窗口数 batch 维度
        x = self.transformer(x) #torch.Size([50, 1, 768])
        x = x.permute(1, 0, 2)  # batch 窗口数 维度

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj #torch.Size([768, 512]) 投射回512维度

        return x #torch.Size([1, 512]) 最后一张图像的信息全部浓缩到这个512维度的向量中


class CLIP(nn.Module):
    """Multi-modal model combining vision and text encoders for joint embeddings based on arxiv.org/abs/2103.00020."""

    def __init__(
        self,
        embed_dim: int, #512
        # vision
        image_resolution: int, #224
        vision_layers: Union[Tuple[int, int, int, int], int], #12
        vision_width: int, #768
        vision_patch_size: int, #32
        # text
        context_length: int, #77
        vocab_size: int, #49408
        transformer_width: int, #512
        transformer_heads: int, #8
        transformer_layers: int, #12
    ):
        """Initializes CLIP model with vision and text components for multi-modal embedding with specified dimensions
        and layers.
        """
        super().__init__()

        self.context_length = context_length
        # vision_layers值是12 说明是一个视觉的transformer框架
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize the parameters of the token and positional embeddings with normal distributions."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features**-0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        """Create a causal attention mask with full attention between vision tokens, using an additive attention mask
        filled with -inf.
        """
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        """Return the data type of the weights of the first convolutional layer in the visual model."""
        return self.visual.conv1.weight.dtype

    def encode_image(self, image): #torch.Size([1, 3, 224, 224])
        """Encodes an input image using the visual model and returns the encoded representation."""
        return self.visual(image.type(self.dtype))

    def encode_text(self, text): #torch.Size([3, 77])
        """Encodes input text using the token embedding and converts it to the specified data type."""
        x = self.token_embedding(text).type(self.dtype)  #torch.Size([3, 77, 512]) 三句话 每句话支持最长长度是77 每个词根编码成512维度的向量
        # ["a diagram", "a dog", "a cat"]
        x = x + self.positional_embedding.type(self.dtype) #torch.Size([77, 512]) 加入
        x = x.permute(1, 0, 2)  # torch.Size([3, 77, 512]) torch.Size([77, 3, 512])
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (end_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection # torch.Size([512, 512])
        # torch.Size([3, 512]) #文本是取出初始相应最大的值
        return x
        # 文本的token化等效于 图片的窗口化
    def forward(self, image, text):
        """Processes input image and text data through encoder modules and returns the respective features."""
        image_features = self.encode_image(image) #torch.Size([1, 3, 224, 224]) -> torch.Size([1, 512])
        text_features = self.encode_text(text) #torch.Size([3, 77])->torch.Size([3, 512])

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t() #torch.Size([1, 3])
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text ##torch.Size([1, 3]) #torch.Size([3, 1])


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16."""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    """Builds and returns a CLIP model from the provided state dictionary."""
    vit = "visual.proj" in state_dict

    if vit: #torch.Size([768, 3, 32, 32])
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")]) #12
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1] #32
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5) #7
        image_resolution = vision_patch_size * grid_size #224
    else:
        counts: list = [
            len({k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")}) for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1] #torch.Size([512, 512])
    context_length = state_dict["positional_embedding"].shape[0] #torch.Size([77, 512])
    vocab_size = state_dict["token_embedding.weight"].shape[0] #torch.Size([49408, 512])
    transformer_width = state_dict["ln_final.weight"].shape[0] #torch.Size([512])
    transformer_heads = transformer_width // 64 #8
    transformer_layers = len({k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")}) #12个残差块

    model = CLIP(
        embed_dim, #512
        image_resolution, #224
        vision_layers, #12
        vision_width, #768
        vision_patch_size, #32
        context_length, #77
        vocab_size, #49408
        transformer_width, #512
        transformer_heads, #8
        transformer_layers, #12
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
