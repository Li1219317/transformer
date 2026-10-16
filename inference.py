"""
模型推理脚本
- 加载 train.py 保存的 checkpoint（内含 model_config + 词表）
- 输入英文文本，输出德文翻译
- 支持 贪心搜索（beam_size=1）与 beam search

运行示例：
  python inference.py --checkpoint checkpoints/best.pt --input "A little girl is playing."
  python inference.py --checkpoint checkpoints/best.pt --interactive
"""
import argparse
import os
import torch

# 缓存目录与训练脚本保持一致
_script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(_script_dir, "hf_cache")

from Transformer import Transformer
from config import ModelConfig, InferConfig, PAD_IDX, BOS_IDX, EOS_IDX
from tokenizer import Vocab, tokenize
from utils import make_src_mask, make_tgt_mask


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = Transformer(**model_cfg.to_dict()).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    # 从 checkpoint 恢复词表
    src_vocab = Vocab(ckpt["src_vocab"])
    tgt_vocab = Vocab(ckpt["tgt_vocab"])
    return model, model_cfg, src_vocab, tgt_vocab


def _block_special_tokens(logits):
    # 禁止解码出 PAD / BOS（EOS 保留，用于自然终止）
    logits[:, [PAD_IDX, BOS_IDX]] = float("-inf")
    return logits


@torch.no_grad()
def greedy_decode(model, src_ids, max_len, device):
    src = torch.tensor([src_ids], dtype=torch.long, device=device)   # (1, L)
    src_mask = make_src_mask(src)
    memory = model.encoder(src, src_mask)      # encoder 只算一次，循环内复用
    tgt = torch.full((1, 1), BOS_IDX, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model.decoder(tgt, memory, make_tgt_mask(tgt), src_mask)
        next_id = _block_special_tokens(logits[:, -1]).argmax(-1, keepdim=True)
        tgt = torch.cat([tgt, next_id], dim=1)
        if next_id.item() == EOS_IDX:
            break
    return tgt[0, 1:].tolist()                 # 去掉 BOS，末尾含 EOS


@torch.no_grad()
def beam_search(model, src_ids, beam_size, max_len, length_penalty, device):
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = make_src_mask(src)
    memory = model.encoder(src, src_mask)

    beams = [(0.0, torch.full((1, 1), BOS_IDX, dtype=torch.long, device=device))]
    finished = []
    for _ in range(max_len):
        candidates = []
        for score, tgt in beams:
            if tgt[0, -1].item() == EOS_IDX:
                finished.append((score, tgt))
                continue
            logits = model.decoder(tgt, memory, make_tgt_mask(tgt), src_mask)
            logp = torch.log_softmax(_block_special_tokens(logits[:, -1]), dim=-1)
            topv, topi = logp.topk(beam_size, dim=-1)
            for v, i in zip(topv[0].tolist(), topi[0].tolist()):
                candidates.append((score + v,
                                   torch.cat([tgt, torch.tensor([[i]], device=device)], dim=1)))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]

    finished += beams
    # 用长度惩罚归一化，避免 beam search 偏好短序列
    best_score, best_seq = max(finished, key=lambda x: x[0] / (x[1].size(1) ** length_penalty))
    return best_seq[0, 1:].tolist()


def translate(model, cfg, src_vocab, tgt_vocab, text):
    device = torch.device(cfg.device)
    src_ids = src_vocab.encode(tokenize(text), max_len=cfg.max_gen_len)
    if cfg.beam_size <= 1:
        out_ids = greedy_decode(model, src_ids, cfg.max_gen_len, device)
    else:
        out_ids = beam_search(model, src_ids, cfg.beam_size, cfg.max_gen_len,
                              cfg.length_penalty, device)
    return tgt_vocab.decode(out_ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str)
    p.add_argument("--input", type=str, help="输入英文文本，如 'A little girl is playing.'")
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--beam_size", type=int)
    p.add_argument("--max_gen_len", type=int)
    p.add_argument("--device", type=str)
    args = p.parse_args()

    cfg = InferConfig()
    for k, v in vars(args).items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)

    model, model_cfg, src_vocab, tgt_vocab = load_model(cfg.checkpoint, torch.device(cfg.device))
    print(f"已加载 {cfg.checkpoint}")
    print(f"词表: en={len(src_vocab)}, de={len(tgt_vocab)}")

    if cfg.interactive:
        print("交互模式：输入英文句子（q 退出）")
        while True:
            text = input("en > ").strip()
            if text in ("q", "quit", "exit"):
                break
            if not text:
                continue
            result = translate(model, cfg, src_vocab, tgt_vocab, text)
            print(f"de < {result}")
    else:
        result = translate(model, cfg, src_vocab, tgt_vocab, cfg.input)
        print(f"en > {cfg.input}")
        print(f"de < {result}")


if __name__ == "__main__":
    main()
