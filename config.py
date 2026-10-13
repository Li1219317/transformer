"""
训练 / 推理参数配置脚本
基于 Transformer.py 中的 Encoder-Decoder 架构，所有可调参数集中在此，
train.py 与 inference.py 统一从这里读取，保证训练与推理配置一致。
"""
from dataclasses import dataclass, asdict, field

import torch

# 特殊 token 约定（数据处理流程统一遵守）
PAD_IDX = 0   # 填充符
BOS_IDX = 1   # 序列起始符
EOS_IDX = 2   # 序列结束符
# 实际 token id 从 3 开始


def _default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelConfig:
    """与 Transformer.__init__ 的参数一一对应"""
    src_vocab_size: int = 8000   # 源语言词表上限（实际大小由 build_vocab 决定）
    tgt_vocab_size: int = 8000   # 目标语言词表上限
    d_model: int = 128           # 隐藏维度（必须能被 num_heads 整除）
    d_ff: int = 512              # FFN 中间层维度
    num_heads: int = 8           # 注意力头数
    num_layers: int = 3          # Encoder / Decoder 层数
    dropout: float = 0.1         # 全局 dropout 概率
    max_len: int = 64            # 预期最大序列长度

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    # 数据（默认合成任务：reverse，把源序列倒序作为目标序列）
    num_train: int = 8000
    num_val: int = 1000
    min_src_len: int = 4
    max_src_len: int = 12
    # 训练超参
    batch_size: int = 64
    epochs: int = 30
    lr: float = 5e-4             # 峰值学习率（warmup 结束时达到）
    warmup_steps: int = 400      # 预热步数，之后按 1/sqrt(step) 衰减
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    # 日志与保存
    device: str = field(default_factory=_default_device)
    save_dir: str = "checkpoints"
    seed: int = 42


@dataclass
class InferConfig:
    checkpoint: str = "checkpoints/best.pt"   # 训练保存的权重文件
    input: str = ""                           # 形如 "5 6 7 8" 的输入
    interactive: bool = False                 # 交互式模式
    beam_size: int = 4                        # 1 = 贪心搜索，>1 = beam search
    max_gen_len: int = 32                     # 解码最大生成长度
    length_penalty: float = 0.7               # beam search 长度惩罚
    device: str = field(default_factory=_default_device)
