"""
下载 Multi30K 机器翻译数据集（en-de）
数据集将缓存到本脚本所在目录的 hf_cache/ 下，避免占用系统盘。
"""
import os

# 必须在 import datasets 之前设置缓存目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["HF_HOME"] = os.path.join(script_dir, "hf_cache")

# 国内网络可启用镜像（取消下一行注释即可）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset


def main():
    print(f"缓存目录: {os.environ['HF_HOME']}")
    print("正在下载/加载 Multi30K (en-de) ...")

    dataset = load_dataset("bentrevett/multi30k")

    print("\n数据集概览:")
    print(dataset)
    print("\n第 1 条训练样本:")
    print(dataset["train"][0])

    # 各 split 统计
    for split_name, split in dataset.items():
        print(f"\n{split_name}: {len(split)} 条")

    # 展示几条样本
    print("\n前 3 条训练样本:")
    for i in range(3):
        item = dataset["train"][i]
        print(f"  [{i}] en: {item['en']}")
        print(f"      de: {item['de']}")

    print("\n下载完成，数据已缓存。")


if __name__ == "__main__":
    main()
