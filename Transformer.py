#这个脚本是transformer模型的实现
import torch
import torch.nn as nn
import math#数学库
#自注意力机制
class SelfAttention(nn.Module):
    def __init__(self,dropout=0.1):#初始化
#dropout是dropout概率
        super(SelfAttention, self).__init__()
#"以 SelfAttention 为起点，在 self 所属对象的方法解析顺序（MRO）中，找到 SelfAttention 之后的那个类。
        self.dropout = nn.Dropout(dropout)#定义成员变量，成员里的dropout就是torch.nn库里的，参数用定义的0.1
        self.softmax = nn.Softmax(dim=-1)#Softmax 把一组实数变成概率分布（每个值在 0~1 之间，且总和为 1）：
    def forward(self, Q, K, V,mask=None):#定义前向传播函数，传入的参数是经过变换的KQV矩阵，掩码默认不开启
        d_k = Q.size()[-1]#确定参数的    维度，就是一个token被转换成的参数向量里有几个参数，最开始是512个
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)#计算自己和其他词汇的相关性，询问是Q，关键词是K，K进行转置之后才能相乘
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        return out,attn
#自注意力机制做的事情就是算出关系，之后给原句子赋予权重
#math.sqrt(d_k)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)  # 输出投影

        self.attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)


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
        return out, attn

import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.2):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)   # 升维
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)   # 降维回 d_model
        self.dropout = nn.Dropout(dropout)
        # 注意：不再在这里定义 self.norm，残差+Norm 由 EncoderLayer/DecoderLayer 负责

    def forward(self, x):
        out = self.fc1(x)          # 升维
        out = F.relu(out)          # 激活
        out = self.dropout(out)
        out = self.fc2(out)        # 降回 d_model
        return out                 #  只做纯变换，不加残差、不做 Norm

class EncoderLayer(nn.Module):#这一部分是编码器
    def __init__(self, d_model, d_ff, num_heads, dropout=0.2):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # 子层1：多头自注意力 + 残差 + Norm
        out, attn = self.attention(src, src, src, mask)# 函数输出了两个结果然后用out和attn接受
        out = self.norm1(src + self.dropout(out))#先将输出结果随机失去活力一部分，之后和自己的输入连接，整体层归一化

        # 子层2：FFN + 残差 + Norm
        out = self.norm2(out + self.dropout(self.feedforward(out)))
        #这是输出的结果先前馈神经网咯，之后随机失活，之后在和自己相加再层归一化？
        return out, attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout=0.2):
        super().__init__()
        # 子层1：自注意力
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        # 子层2：交叉注意力
        self.crossattention = MultiHeadAttention(d_model, num_heads, dropout)
        # 子层3：前馈网络
        self.feedforward = FeedForward(d_model, d_ff, dropout)

        # 三个 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 子层1：自注意力 + 残差 + Norm
        out, _ = self.attention(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(out))

        # 子层2：交叉注意力（Q=tgt, K=V=memory）+ 残差 + Norm
        out, _ = self.crossattention(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(out))

        # 子层3：FFN + 残差 + Norm
        out = self.norm3(tgt + self.dropout(self.feedforward(tgt)))

        return out
#我预先算好了一堆位置的编码，现在输入了多少个 token（seq_len），就从里面取前 seq_len 个；这些位置编码通过广播复制到 batch 里每一句话上，再和我每个 token 的向量逐元素相加，把位置信息注入进去。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model) #创建了一张位置编码的词表，先全都填0，大小是5000*512， 随便定的token数目和对应一个    token 的参数两
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维
        pe = pe.unsqueeze(0)                            # [1, max_len, d_model]

        self.register_buffer('pe', pe)                  # 不参与训练，但随模型保存

    def forward(self, x):
        seq_len = x.size(1)#先取出我的向量有几个token
        x = x + self.pe[:, :seq_len, :]            #之后将向量和位置词表的对应元素     # 广播相加
        return self.dropout(x)          #随机失活


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_layers, dropout=0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 1. 词嵌入 + 缩放
        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)

        # 2. 加位置编码 + dropout
        x = self.pos_encoder(x)

        # 3. 逐层通过 EncoderLayer
        for layer in self.layers:
            x, _ = layer(x, src_mask)

        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_layers, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        out = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        out = self.pos_encoder(out)

        for layer in self.layers:
            out = layer(out, memory, tgt_mask, memory_mask)   #  只接 1 个

        return self.fc_out(out)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, d_ff=2048, num_heads=8,
                 num_layers=6, dropout=0.1, max_len=5000):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, d_ff, num_heads, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, d_ff, num_heads, num_layers, dropout)
        # 注意：Encoder 和 Decoder 的 vocab_size 可以不同（源语言/目标语言）

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)                       # 编码源语言
        out = self.decoder(tgt, memory, tgt_mask, memory_mask)      # 解码目标语言
        return out                                                  # [B, tgt_len, tgt_vocab]