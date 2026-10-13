"""
分词与词表模块
- 分词：空格分词 + 标点分离（英文/德文通用）
- 词表：按词频截断，特殊 token 固定为 PAD/BOS/EOS/UNK
- 编码：文本 -> token id 列表；解码：id 列表 -> 文本
"""
import re
from collections import Counter
from typing import List, Dict

from config import PAD_IDX, BOS_IDX, EOS_IDX

UNK_IDX = 3   # 未登录词

# 把标点与单词分离的正则：在标点前后加空格
_TOKENIZE_RE = re.compile(r"([.,!?;:()\"'-])")


def tokenize(text: str) -> List[str]:
    """简单分词：小写化 + 标点分离 + 按空格切分"""
    text = text.lower().strip()
    text = _TOKENIZE_RE.sub(r" \1 ", text)
    return [t for t in text.split() if t]


class Vocab:
    def __init__(self, token2id: Dict[str, int]):
        self.token2id = token2id
        self.id2token = {v: k for k, v in token2id.items()}

    def __len__(self):
        return len(self.token2id)

    def encode(self, tokens: List[str], max_len: int = None) -> List[int]:
        """token 列表 -> id 列表（不含 BOS/EOS，调用方自行添加）"""
        ids = [self.token2id.get(t, UNK_IDX) for t in tokens]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int], skip_special=True) -> str:
        """id 列表 -> 文本"""
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, "<unk>")
            if skip_special and tok in ("<pad>", "<bos>", "<eos>", "<unk>"):
                continue
            tokens.append(tok)
        # 把分离的标点拼回去
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text


def build_vocab(sentences: List[str], max_size: int = 8000, min_freq: int = 2) -> Vocab:
    """从句子列表构建词表，按词频截断"""
    counter = Counter()
    for s in sentences:
        counter.update(tokenize(s))

    # 特殊 token 固定位置
    token2id = {"<pad>": PAD_IDX, "<bos>": BOS_IDX, "<eos>": EOS_IDX, "<unk>": UNK_IDX}
    # 按词频从高到低，过滤低频词，截断到 max_size
    for tok, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(token2id) >= max_size:
            break
        if tok not in token2id:
            token2id[tok] = len(token2id)
    return Vocab(token2id)
