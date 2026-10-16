"""
训练工具模块：可复用的训练 / 评估 / 保存逻辑
- set_seed：统一设置随机种子，保证可复现
- train_one_epoch：单轮训练
- evaluate：验证集评估
- save_checkpoint：保存 checkpoint
"""
import random

import torch

from utils import make_src_mask, make_tgt_mask


def set_seed(seed):
    # 设置 Python 随机数种子
    random.seed(seed)
    # 设置 PyTorch CPU 随机数生成器的种子，影响权重初始化
    torch.manual_seed(seed)
    # 设置所有 GPU 上的随机数种子
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, cfg):
    # 切换到训练模式（启用 dropout 等）
    model.train()
    # 累计损失与批次数
    total, n = 0.0, 0
    for src, tgt_in, tgt_out in loader:
        # 将数据搬运到目标设备
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        # 构造源序列 padding 掩码
        src_mask = make_src_mask(src)
        # 构造目标序列因果 + padding 掩码
        tgt_mask = make_tgt_mask(tgt_in)
        # memory_mask 传 src_mask：交叉注意力同样要屏蔽 encoder 输出的 padding
        logits = model(src, tgt_in, src_mask, tgt_mask, src_mask)   # (B, T, V)
        # 计算交叉熵损失（展平后按 token 计算）
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        # 清空梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # 更新参数
        optimizer.step()
        # 更新学习率
        scheduler.step()
        # 累计损失
        total += loss.item()
        n += 1
    # 返回该 epoch 的平均损失
    return total / max(n, 1)


@torch.no_grad()   # 评估阶段禁用梯度计算，节省显存
def evaluate(model, loader, criterion, device):
    # 切换到评估模式（关闭 dropout 等）
    model.eval()
    # 累计损失与批次数
    total, n = 0.0, 0
    for src, tgt_in, tgt_out in loader:
        # 将数据搬运到目标设备
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        # 构造源序列 padding 掩码
        src_mask = make_src_mask(src)
        # 构造目标序列因果 + padding 掩码
        tgt_mask = make_tgt_mask(tgt_in)
        # 前向计算 logits
        logits = model(src, tgt_in, src_mask, tgt_mask, src_mask)
        # 计算交叉熵损失
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        # 累计损失
        total += loss.item()
        n += 1
    # 返回平均损失
    return total / max(n, 1)


def save_checkpoint(path, model, optimizer, model_cfg, src_vocab, tgt_vocab, epoch, val_loss):
    # 打包模型、优化器、配置、词表等状态，用于保存 checkpoint
    state = {"model": model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "model_config": model_cfg.to_dict(),
             "src_vocab": src_vocab.token2id,
             "tgt_vocab": tgt_vocab.token2id,
             "epoch": epoch, "val_loss": val_loss}
    # 保存到指定路径
    torch.save(state, path)
