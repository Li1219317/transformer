"""
通用工具模块：与具体任务无关的可复用代码块
- 掩码工具：make_src_mask / make_tgt_mask（训练与推理脚本共用）
- 批处理：collate_fn（变长序列右侧 PAD 对齐）
- 学习率调度：lr_lambda（线性预热 + 逆平方根衰减）
"""
import torch

from config import PAD_IDX


def make_src_mask(src, pad_idx=PAD_IDX):
    # (B, L) -> (B, 1, 1, L)：屏蔽源序列 padding，可广播到所有注意力头
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, pad_idx=PAD_IDX):
    # (B, L) -> (B, 1, L, L)：因果掩码（下三角）+ padding 掩码
    # 目标序列长度
    L = tgt.size(1)
    # 构造下三角因果矩阵，防止解码时看到未来位置
    causal = torch.tril(torch.ones(L, L, device=tgt.device, dtype=torch.bool))
    causal = causal.unsqueeze(0).unsqueeze(0)              # (1, 1, L, L)
    pad = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)       # (B, 1, 1, L)
    # 按位与：同时屏蔽 padding 与未来位置
    return causal & pad                                    # (B, 1, L, L)


def collate_fn(batch):
    # 同批变长序列右侧 PAD 对齐
    def pad(seqs):
        # 本批中的最大序列长度
        m = max(len(s) for s in seqs)
        # 将每条序列补齐到长度 m 并转为张量
        return torch.tensor([s + [PAD_IDX] * (m - len(s)) for s in seqs], dtype=torch.long)
    # 将 batch 拆分为源序列、解码输入、解码目标三组
    srcs, tgt_ins, tgt_outs = zip(*batch)
    # 分别对齐后返回
    return pad(srcs), pad(tgt_ins), pad(tgt_outs)


def lr_lambda(step, warmup):
    # 步数下限设为 1，避免除零
    step = max(step, 1)
    # 预热阶段：学习率随步数线性增长
    if step < warmup:
        return step / warmup
    # 预热之后：按步数的逆平方根衰减
    return (warmup / step) ** 0.5
