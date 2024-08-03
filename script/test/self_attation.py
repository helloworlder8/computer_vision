import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads): #256 8
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads #32

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 定义权重矩阵
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        # 分割输入为多个头
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 分割embedding
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 乘法注意力分数计算
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # 缩放点积注意力
        attention = attention / (self.embed_size ** (1 / 2))
        attention = torch.softmax(attention, dim=-1)

        # 注意力机制后的值
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # 最后通过一个线性层
        out = self.fc_out(out)
        return out

# 假设的参数
embed_size = 256
heads = 8
attention = SelfAttention(embed_size, heads)

# 随机生成一些数据来测试我们的自注意力机制
x = torch.randn((1, 10, embed_size))
output = attention(x, x, x)  # 自注意力机制通常用相同的输入作为values, keys, queries
print(output.shape)  # 查看输出形状
