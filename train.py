"""
模型训练脚本
- 模型：Transformer.py 中的 Encoder-Decoder Transformer
- 数据：Multi30K (en -> de) 机器翻译数据集，自动从 Hugging Face 下载并缓存到脚本目录

运行示例：
  python train.py
  python train.py --epochs 30 --batch_size 64 --device cpu
"""
import argparse          # 命令行参数解析
import os                # 路径与目录操作
import time              # 计时与时间戳

# 在 import datasets 之前设置缓存目录到脚本所在目录
# 当前脚本所在目录
_script_dir = os.path.dirname(os.path.abspath(__file__))
# 将 Hugging Face 缓存目录指向脚本目录下的 hf_cache
os.environ["HF_HOME"] = os.path.join(_script_dir, "hf_cache")
# 国内网络可取消下一行注释启用镜像
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch                                        # PyTorch 核心库
import torch.nn as nn                               # 神经网络模块（损失函数等）
from torch.utils.data import Dataset, DataLoader    # 数据集与批加载器
from torch.utils.tensorboard import SummaryWriter   # TensorBoard 日志记录
from datasets import load_dataset                   # Hugging Face 数据集加载

from Transformer import Transformer                                     # 本地 Transformer 模型
from config import ModelConfig, TrainConfig, PAD_IDX, BOS_IDX, EOS_IDX  # 配置类与特殊 token
from tokenizer import build_vocab, Vocab, UNK_IDX                       # 分词与词表工具
from utils import collate_fn, lr_lambda                                 # 通用可复用工具
from train_utils import set_seed, train_one_epoch, evaluate, save_checkpoint  # 训练相关工具


# ---------------------- 数据：Multi30K (en -> de) ----------------------
class TranslationDataset(Dataset):
    """已编码为 id 的翻译数据集"""
    def __init__(self, src_ids_list, tgt_ids_list):
        # 源语言 id 序列列表
        self.src_ids = src_ids_list
        # 目标语言 id 序列列表
        self.tgt_ids = tgt_ids_list

    def __len__(self):
        # 返回样本数量
        return len(self.src_ids)

    def __getitem__(self, idx):
        # 取第 idx 条源/目标 id 序列
        src = self.src_ids[idx]
        tgt = self.tgt_ids[idx]
        tgt_in = [BOS_IDX] + tgt       # decoder 输入：<BOS> + 目标
        tgt_out = tgt + [EOS_IDX]      # decoder 目标：目标 + <EOS>
        return src, tgt_in, tgt_out


def encode_sentences(vocab: Vocab, sentences, max_len):
    """把文本列表编码为 id 列表列表，截断到 max_len"""
    # 延迟导入分词函数，避免循环依赖
    from tokenizer import tokenize
    # 保存编码结果
    result = []
    for s in sentences:
        # 先分词再编码为 id，并按 max_len 截断
        ids = vocab.encode(tokenize(s), max_len=max_len)
        result.append(ids)
    return result


def get_dataloaders(cfg, model_cfg, src_vocab, tgt_vocab):
    # 提示正在加载数据集
    print("加载 Multi30K (en-de) 数据集...")
    # 从 Hugging Face 加载 Multi30K 数据集
    dataset = load_dataset("bentrevett/multi30k")

    # 源序列最大长度
    max_src = model_cfg.max_len
    # 目标序列最大长度
    max_tgt = model_cfg.max_len

    def build_split(split):
        # 编码该 split 的英文句子
        src_ids = encode_sentences(src_vocab, dataset[split]["en"], max_src)
        # 编码该 split 的德文句子
        tgt_ids = encode_sentences(tgt_vocab, dataset[split]["de"], max_tgt)
        # 返回编码后的数据集对象
        return TranslationDataset(src_ids, tgt_ids)

    # 构建训练集
    train_ds = build_split("train")
    # 构建验证集
    val_ds = build_split("validation")

    # 训练集 DataLoader：打乱顺序
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          collate_fn=collate_fn)
    # 验证集 DataLoader：不打乱顺序
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_fn)
    return train_dl, val_dl


# ---------------------- 主流程 ----------------------
def parse_args():
    # 建立参数解析器
    p = argparse.ArgumentParser()
    # 定义参数
    p.add_argument("--epochs", type=int, help="覆盖训练轮数")
    p.add_argument("--batch_size", type=int, help="覆盖批大小")
    p.add_argument("--device", type=str, help="cuda / cpu")
    p.add_argument("--save_dir", type=str, help="checkpoint 保存目录")
    # 解析命令
    return p.parse_args()


def main():
    # 拿到命令行参数
    args = parse_args()
    # 建立模型配置，将写好的模型配置转为配置类对象
    model_cfg = ModelConfig()
    # 建立训练配置
    cfg = TrainConfig()
    # 覆盖原有配置
    for k, v in vars(args).items():
        # 仅覆盖命令行显式传入且 TrainConfig 中存在的字段
        if v is not None and hasattr(cfg, k):
            # 动态设置属性，实现配置覆盖
            setattr(cfg, k, v)
    # 统一设置随机种子，保证结果可复现
    set_seed(cfg.seed)
    # 把配置里的字符串（如"cuda"、"cpu"、"cuda:0"）转成PyTorch的device对象。
    device = torch.device(cfg.device)
    # 创建 checkpoint 保存目录（已存在则跳过）
    os.makedirs(cfg.save_dir, exist_ok=True)

    # TensorBoard 日志目录：runs/时间戳
    log_dir = os.path.join(_script_dir, "runs", time.strftime("%Y%m%d_%H%M%S"))
    # 初始化 TensorBoard 写入器，用于记录训练曲线
    writer = SummaryWriter(log_dir=log_dir)
    # 打印日志目录便于查看
    print(f"TensorBoard 日志: {log_dir}")

    # 先加载原始数据，用于构建词表
    print("加载 Multi30K 构建词表...")
    # 从 Hugging Face 下载并加载 Multi30K 数据集
    dataset = load_dataset("bentrevett/multi30k")
    # 基于英文训练集构建源语言词表，限制最大词表大小
    src_vocab = build_vocab(dataset["train"]["en"], max_size=model_cfg.src_vocab_size)
    # 基于德文训练集构建目标语言词表，限制最大词表大小
    tgt_vocab = build_vocab(dataset["train"]["de"], max_size=model_cfg.tgt_vocab_size)
    # 词表实际大小作为模型 vocab_size（包含特殊 token）
    model_cfg.src_vocab_size = len(src_vocab)
    model_cfg.tgt_vocab_size = len(tgt_vocab)
    # 打印两个词表的实际大小，便于核对
    print(f"源(en)词表: {len(src_vocab)} | 目标(de)词表: {len(tgt_vocab)}")

    # 构建训练集与验证集的 DataLoader
    train_dl, val_dl = get_dataloaders(cfg, model_cfg, src_vocab, tgt_vocab)
    # 实例化 Transformer 模型并放到目标设备上
    model = Transformer(**model_cfg.to_dict()).to(device)
    # 统计模型总参数量
    n_params = sum(p.numel() for p in model.parameters())
    # 打印模型参数量与所用设备
    print(f"模型参数量: {n_params / 1e6:.2f}M | device: {device}")
    # 打印最终模型配置，便于核对
    print(f"模型配置: {model_cfg.to_dict()}")

    # 交叉熵损失，忽略 padding 位置，启用 label smoothing
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=cfg.label_smoothing)
    # Adam 优化器，使用 Transformer 论文推荐的 betas 与 eps
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    # 学习率调度：线性预热 + 逆平方根衰减
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_lambda(s, cfg.warmup_steps))

    # 记录历史最优验证损失，初始设为正无穷
    best_val = float("inf")
    # 训练循环：逐 epoch 训练并验证
    for epoch in range(1, cfg.epochs + 1):
        # 记录当前 epoch 起始时间，用于统计耗时
        t0 = time.time()
        # 训练一个 epoch，返回训练集平均损失
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, scheduler, device, cfg)
        # 在验证集上评估，返回验证集平均损失
        val_loss = evaluate(model, val_dl, criterion, device)
        # 取当前学习率用于日志输出
        lr_now = optimizer.param_groups[0]["lr"]
        # 打印本 epoch 的训练/验证损失、学习率与耗时
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| lr {lr_now:.2e} | {time.time() - t0:.1f}s")

        # 记录到 TensorBoard
        # 写入训练损失曲线
        writer.add_scalar("Loss/train", train_loss, epoch)
        # 写入验证损失曲线
        writer.add_scalar("Loss/val", val_loss, epoch)
        # 写入当前学习率
        writer.add_scalar("LR", lr_now, epoch)

        # 每个 epoch 都保存为 last.pt（最新检查点）
        save_checkpoint(os.path.join(cfg.save_dir, "last.pt"),
                        model, optimizer, model_cfg, src_vocab, tgt_vocab, epoch, val_loss)
        # 当验证损失刷新历史最优时保存 best.pt
        if val_loss < best_val:
            # 更新历史最优验证损失
            best_val = val_loss
            # 保存当前最优模型
            save_checkpoint(os.path.join(cfg.save_dir, "best.pt"),
                            model, optimizer, model_cfg, src_vocab, tgt_vocab, epoch, val_loss)
            # 提示已保存最优模型
            print(f"  -> 保存最优模型 (val {best_val:.4f})")

    # 训练结束，打印最优验证损失
    print(f"训练完成，最优 val loss = {best_val:.4f}")
    # 关闭 TensorBoard 写入器，刷新缓冲
    writer.close()


# 作为脚本直接运行时执行训练流程
if __name__ == "__main__":
    main()
