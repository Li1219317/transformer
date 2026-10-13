"""
模型训练脚本
- 模型：Transformer.py 中的 Encoder-Decoder Transformer
- 数据：Multi30K (en -> de) 机器翻译数据集，自动从 Hugging Face 下载并缓存到脚本目录

运行示例：
  python train.py
  python train.py --epochs 30 --batch_size 64 --device cpu
"""
import argparse
import os
import random
import time

# 在 import datasets 之前设置缓存目录到脚本所在目录
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(_script_dir, "hf_cache")
# 国内网络可取消下一行注释启用镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from Transformer import Transformer
from config import ModelConfig, TrainConfig, PAD_IDX, BOS_IDX, EOS_IDX
from tokenizer import build_vocab, Vocab, UNK_IDX


# ---------------------- 掩码工具（推理脚本也会复用） ----------------------
def make_src_mask(src, pad_idx=PAD_IDX):
    # (B, L) -> (B, 1, 1, L)：屏蔽源序列 padding，可广播到所有注意力头
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt, pad_idx=PAD_IDX):
    # (B, L) -> (B, 1, L, L)：因果掩码（下三角）+ padding 掩码
    L = tgt.size(1)
    causal = torch.tril(torch.ones(L, L, device=tgt.device, dtype=torch.bool))
    causal = causal.unsqueeze(0).unsqueeze(0)              # (1, 1, L, L)
    pad = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)       # (B, 1, 1, L)
    return causal & pad                                    # (B, 1, L, L)


# ---------------------- 数据：Multi30K (en -> de) ----------------------
class TranslationDataset(Dataset):
    """已编码为 id 的翻译数据集"""
    def __init__(self, src_ids_list, tgt_ids_list):
        self.src_ids = src_ids_list
        self.tgt_ids = tgt_ids_list

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        src = self.src_ids[idx]
        tgt = self.tgt_ids[idx]
        tgt_in = [BOS_IDX] + tgt       # decoder 输入：<BOS> + 目标
        tgt_out = tgt + [EOS_IDX]      # decoder 目标：目标 + <EOS>
        return src, tgt_in, tgt_out


def collate_fn(batch):
    # 同批变长序列右侧 PAD 对齐
    def pad(seqs):
        m = max(len(s) for s in seqs)
        return torch.tensor([s + [PAD_IDX] * (m - len(s)) for s in seqs], dtype=torch.long)
    srcs, tgt_ins, tgt_outs = zip(*batch)
    return pad(srcs), pad(tgt_ins), pad(tgt_outs)


def encode_sentences(vocab: Vocab, sentences, max_len):
    """把文本列表编码为 id 列表列表，截断到 max_len"""
    from tokenizer import tokenize
    result = []
    for s in sentences:
        ids = vocab.encode(tokenize(s), max_len=max_len)
        result.append(ids)
    return result


def get_dataloaders(cfg, model_cfg, src_vocab, tgt_vocab):
    print("加载 Multi30K (en-de) 数据集...")
    dataset = load_dataset("bentrevett/multi30k")

    max_src = model_cfg.max_len
    max_tgt = model_cfg.max_len

    def build_split(split):
        src_ids = encode_sentences(src_vocab, dataset[split]["en"], max_src)
        tgt_ids = encode_sentences(tgt_vocab, dataset[split]["de"], max_tgt)
        return TranslationDataset(src_ids, tgt_ids)

    train_ds = build_split("train")
    val_ds = build_split("validation")

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_fn)
    return train_dl, val_dl


# ---------------------- 学习率调度（线性预热 + 逆平方根衰减） ----------------------
def lr_lambda(step, warmup):
    step = max(step, 1)
    if step < warmup:
        return step / warmup
    return (warmup / step) ** 0.5


# ---------------------- 训练 / 验证 ----------------------
def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, cfg):
    model.train()
    total, n = 0.0, 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)
        # memory_mask 传 src_mask：交叉注意力同样要屏蔽 encoder 输出的 padding
        logits = model(src, tgt_in, src_mask, tgt_mask, src_mask)   # (B, T, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)
        logits = model(src, tgt_in, src_mask, tgt_mask, src_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total += loss.item()
        n += 1
    return total / max(n, 1)


# ---------------------- 主流程 ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, help="覆盖训练轮数")
    p.add_argument("--batch_size", type=int, help="覆盖批大小")
    p.add_argument("--device", type=str, help="cuda / cpu")
    p.add_argument("--save_dir", type=str, help="checkpoint 保存目录")
    return p.parse_args()


def main():
    args = parse_args()
    model_cfg = ModelConfig()
    cfg = TrainConfig()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # TensorBoard 日志目录：runs/时间戳
    log_dir = os.path.join(_script_dir, "runs", time.strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard 日志: {log_dir}")

    # 先加载原始数据，用于构建词表
    print("加载 Multi30K 构建词表...")
    dataset = load_dataset("bentrevett/multi30k")
    src_vocab = build_vocab(dataset["train"]["en"], max_size=model_cfg.src_vocab_size)
    tgt_vocab = build_vocab(dataset["train"]["de"], max_size=model_cfg.tgt_vocab_size)
    # 词表实际大小作为模型 vocab_size（包含特殊 token）
    model_cfg.src_vocab_size = len(src_vocab)
    model_cfg.tgt_vocab_size = len(tgt_vocab)
    print(f"源(en)词表: {len(src_vocab)} | 目标(de)词表: {len(tgt_vocab)}")

    train_dl, val_dl = get_dataloaders(cfg, model_cfg, src_vocab, tgt_vocab)
    model = Transformer(**model_cfg.to_dict()).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params / 1e6:.2f}M | device: {device}")
    print(f"模型配置: {model_cfg.to_dict()}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg.warmup_steps))

    best_val = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device, cfg)
        val_loss = evaluate(model, val_dl, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| lr {lr_now:.2e} | {time.time() - t0:.1f}s")

        # 记录到 TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR", lr_now, epoch)

        state = {"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "model_config": model_cfg.to_dict(),
                 "src_vocab": src_vocab.token2id,
                 "tgt_vocab": tgt_vocab.token2id,
                 "epoch": epoch, "val_loss": val_loss}
        torch.save(state, os.path.join(cfg.save_dir, "last.pt"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(state, os.path.join(cfg.save_dir, "best.pt"))
            print(f"  -> 保存最优模型 (val {best_val:.4f})")

    print(f"训练完成，最优 val loss = {best_val:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
