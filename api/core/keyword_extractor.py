# /core/keyword_extractor.py (Real TF-based version)

import jieba
from collections import Counter
from typing import List, Dict

from utils.stopwords_loader import load_stopwords


def get_keywords_from_engine(text: str, top_k: int = 5) -> List[Dict]:
    """
    一个基于词频(TF)的简易关键词提取引擎。

    :param text: 输入的文本字符串。
    :param top_k: 返回权重最高的关键词数量。
    :return: 一个包含关键词和其权重的字典列表。
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # 1. 加载停用词
    stopwords = load_stopwords()

    # 2. 使用jieba进行分词
    words = jieba.lcut(text)

    # 3. 过滤掉停用词和单个字的词
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]

    # 4. 统计词频
    word_counts = Counter(filtered_words)

    # 5. 获取频率最高的 top_k 个词
    top_words = word_counts.most_common(top_k)

    # 6. 格式化成API需要的输出格式
    # 这里的 'weight' 就是词的频次，简单直观
    keywords = [{"keyword": word, "weight": count} for word, count in top_words]

    print(f"核心逻辑层: 提取到关键词: {keywords}")
    return keywords
