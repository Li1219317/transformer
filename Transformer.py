import torch
import torch.nn as nn
import math

from numpy.ma.core import transpose


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, Q, K, V,mask=None):
        d_k = Q.size()[-1]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out,attn
#自注意力机制做的事情就是算出关系，之后给原句子赋予权重

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)  # 输出投影

        self.attention = SelfAttention()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 1. 线性变换 + 拆多头
        # (B, seq, d_model) -> (B, seq, H, d_k) -> (B, H, seq, d_k)
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 每个头做自注意力
        out, attn = self.attention(Q, K, V, mask=mask)
        # out: (B, H, seq, d_k)

        # 3. 拼回多头
        # (B, H, seq, d_k) -> (B, seq, H, d_k) -> (B, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. 输出投影 + dropout
        out = self.W_O(out)
        out = self.dropout(out)

        # 5. 残差 + LayerNorm（用输入 Q 做残差；注意此处 Q 已被覆盖，需提前保存）
        return out, attn

import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.2):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)   # 升维,用一个全链接层，将512维数的参数增加到更多，使得模型有更多的语义可以学习，一般来讲模型变得更大也是这一步在更大
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)   # 降维回 d_model
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model) #层归一化，有些数值可能非常不均匀，在反向传播的过程中y = kx + b ，定义损失函数之后目的是让损失函数变得更小，可以通过更新参数k 和b来达到，那应该增大还是减小呢？
        #就可以算出来损失函数对k 的倒数。我们可见，这个损失函数里面是有x存在的，算出的梯度也是有x 存在的，所以如果x 的方差太大，那就会造成梯度各种各样
        #假设现在有两个样本A，B 。A算出来的梯度很大，B算出来的很小，那么A下一步就会把权重值变得很大，B想把权重值拽回来也不可能，K一直达不到最优的使得损失函数很小的值，就会不收敛

    def forward(self, x):
        out = self.fc1(x)                 #反向传播更新全连接层的权重 W，使对任务有用的输入维度获得较大的正权重（前向中被放大），无用的维度获得较小或负的权重（前向中被抑制）。最终体现在输出 out 上，就是某些语义被增强、某些被削弱，而 out 本身可正可负
        out = F.relu(out)                 # 激活ReLU(x)=max(0,x)就是让负数变为0 这样就不参与运算了
        out = self.dropout(out)
        out = self.fc2(out)               # 降回 d_model
        return self.norm(out + x)         # 残差 + LayerNorm

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.2):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # 子层1：多头自注意力 + 残差 + Norm
        out, attn = self.attention(src, src, src, mask)
        out = self.norm1(src + self.dropout(out))

        # 子层2：FFN + 残差 + Norm
        out = self.norm2(out + self.dropout(self.feedforward(out)))
        return out, attn
