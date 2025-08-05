# /utils/domain_lexicon_builder.py (Integrated Version)

import pandas as pd
import jieba
from collections import Counter, defaultdict
import math
import re
import argparse
import os
from tqdm import tqdm
from pathlib import Path


# --- 您的新词发现代码 (完全保留) ---

class NewWordDiscoverer:
    """
    一个基于统计方法的中文新词发现工具 (优化版)。
    它通过一次遍历预先计算所有统计量，极大地提升了运行速度。
    """

    def __init__(self, corpus, min_freq=5, min_pmi=1.0, min_entropy=0.5, word_len=2):
        self.corpus = corpus
        self.min_freq = min_freq
        self.min_pmi = min_pmi
        self.min_entropy = min_entropy
        self.word_len = word_len

    @staticmethod
    def _calculate_entropy(neighbor_list):
        """静态方法，用于计算信息熵"""
        if not neighbor_list:
            return 0
        counts = Counter(neighbor_list)
        total = len(neighbor_list)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(p, 2)
        return entropy

    def discover(self, output_path):
        """
        执行新词发现流程并保存结果。
        """
        print("Step 1: 正在进行单次遍历，预计算所有统计信息...")
        word_counts = Counter()
        bigram_counts = Counter()
        left_neighbors = defaultdict(list)
        right_neighbors = defaultdict(list)
        all_docs_words = [jieba.lcut(str(doc)) for doc in self.corpus]
        for doc_words in tqdm(all_docs_words, desc="Processing documents"):
            word_counts.update(doc_words)
            for i in range(len(doc_words) - 1):
                bigram = (doc_words[i], doc_words[i + 1])
                bigram_counts[bigram] += 1
                if i > 0:
                    left_neighbors[bigram].append(doc_words[i - 1])
                if i < len(doc_words) - 2:
                    right_neighbors[bigram].append(doc_words[i + 2])
        total_words = sum(word_counts.values())
        print("\nStep 2: 正在筛选候选词并计算得分...")
        candidates = {}
        for bigram, freq in tqdm(bigram_counts.items(), desc="Filtering candidates"):
            if freq < self.min_freq: continue
            word1, word2 = bigram
            if re.search(r'[^\u4e00-\u9fa5]', word1) or re.search(r'[^\u4e00-\u9fa5]', word2): continue
            p_word1 = word_counts[word1] / total_words
            p_word2 = word_counts[word2] / total_words
            p_bigram = freq / total_words
            if p_word1 == 0 or p_word2 == 0: continue
            pmi = math.log(p_bigram / (p_word1 * p_word2), 2)
            if pmi < self.min_pmi: continue
            left_entropy = self._calculate_entropy(left_neighbors[bigram])
            right_entropy = self._calculate_entropy(right_neighbors[bigram])
            entropy = left_entropy + right_entropy
            if entropy < self.min_entropy: continue
            candidate_word = "".join(bigram)
            candidates[candidate_word] = (freq, pmi, entropy)
        print(f"\nStep 3: 发现 {len(candidates)} 个候选新词。正在排序并保存...")
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 格式: 候选新词 词频 点互信息(凝固度) 左右邻字信息熵(自由度)\n")
            for word, scores in sorted_candidates:
                f.write(f"{word} {scores[0]} {scores[1]:.2f} {scores[2]:.2f}\n")
        print(f"新词发现完成！结果已保存至: {output_path}")
        return sorted_candidates


def build_domain_lexicon(corpus_path, output_path, min_freq=10, min_pmi=1.5, min_entropy=1.0):
    """
    一个完整的、用于构建领域词典的函数。
    """
    print("--- 开始执行【领域词典构建】任务 (优化版) ---")
    df = pd.read_csv(corpus_path)
    # 假设文本列名为 'review' 或 'cleaned_text'
    text_column = 'review' if 'review' in df.columns else 'cleaned_text'
    if text_column not in df.columns:
        raise ValueError(f"Corpus CSV must contain 'review' or 'cleaned_text' column.")
    corpus = df[text_column].dropna()
    discoverer = NewWordDiscoverer(corpus=corpus, min_freq=min_freq, min_pmi=min_pmi, min_entropy=min_entropy)
    discoverer.discover(output_path)
    print("--- 【领域词典构建】任务完成 ---")


# --- 【新增】加载基础情感词典的函数 ---

# 获取项目根目录的绝对路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_base_sentiment_lexicon():
    """
    加载情感分析所需的基础情感词典资源 (正面/负面)。
    """
    lexicon = {}
    print("--- 正在加载基础情感词典 ---")

    try:
        # 根据您的截图，词典在 data/Bosonnlp/情感词典/ 目录下
        pos_path = PROJECT_ROOT / "data" / "Bosonnlp" / "情感词典" / "正面情感词.txt"
        neg_path = PROJECT_ROOT / "data" / "Bosonnlp" / "情感词典" / "负面情感词.txt"

        # 使用集合(set)进行存储，查找速度更快
        with open(pos_path, 'r', encoding='utf-8') as f:
            lexicon['pos'] = {line.strip() for line in f if line.strip()}
        with open(neg_path, 'r', encoding='utf-8') as f:
            lexicon['neg'] = {line.strip() for line in f if line.strip()}

        print(f"正面词典加载完成，共 {len(lexicon['pos'])} 个词。")
        print(f"负面词典加载完成，共 {len(lexicon['neg'])} 个词。")

    except FileNotFoundError as e:
        print(f"【错误】情感词典文件未找到，请检查路径: {e}")
        return None

    print("--- 基础情感词典加载完毕 ---\n")
    return lexicon


# --- 主程序入口 (完全保留) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从给定的语料库中发现新词，并构建领域词典。")
    parser.add_argument('--corpus_path', type=str, default='data/weibo_senti_100k.csv', help="输入语料库的文件路径 (CSV格式)。")
    parser.add_argument('--output_path', type=str, default=None, help="输出的候选新词文件的保存路径。")
    args = parser.parse_args()
    output_path = args.output_path
    if output_path is None:
        corpus_dir = os.path.dirname(args.corpus_path)
        base_name = os.path.basename(args.corpus_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(corpus_dir, f'{name_without_ext}_discovered_words.txt')
        print(f"提示：未指定输出路径，将自动使用默认路径: {output_path}")
    build_domain_lexicon(corpus_path=args.corpus_path, output_path=output_path)
