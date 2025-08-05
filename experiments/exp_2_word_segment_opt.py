# experiments/exp_2_word_segment_opt.py

import pandas as pd
import jieba
import sys
import os

# 将上级目录添加到系统路径中，以便导入utils模块
sys.path.append('..')
from utils.model_evaluator import evaluate_and_visualize


def get_snownlp_dicts():
    """
    一个健壮的函数，用于自动查找并返回SnowNLP自带的情感词典路径。
    """
    try:
        import snownlp
        # snownlp.__file__ 会给出snownlp包中__init__.py文件的路径
        # os.path.dirname() 会获取该文件所在的目录
        package_path = os.path.dirname(snownlp.__file__)
        pos_path = os.path.join(package_path, 'sentiment', 'pos.txt')
        neg_path = os.path.join(package_path, 'sentiment', 'neg.txt')

        if not os.path.exists(pos_path) or not os.path.exists(neg_path):
            raise FileNotFoundError

        print("成功从SnowNLP库中自动定位情感词典。")
        return pos_path, neg_path
    except (ImportError, FileNotFoundError):
        print("警告：无法从SnowNLP自动加载情感词典。")
        print("请确保snownlp已正确安装。程序将退出。")
        sys.exit(1)


def load_sentiment_words(pos_path, neg_path):
    """加载情感词典"""
    positive_words = set()
    negative_words = set()
    try:
        with open(pos_path, 'r', encoding='utf-8') as f:
            for word in f:
                positive_words.add(word.strip())
        with open(neg_path, 'r', encoding='utf-8') as f:
            for word in f:
                negative_words.add(word.strip())
    except FileNotFoundError as e:
        print(f"错误：找不到情感词典文件: {e.filename}")
        print("程序将退出。请检查词典路径是否正确。")
        sys.exit(1)
    return positive_words, negative_words


def custom_sentiment_predict(text, positive_words, negative_words):
    """
    基于Jieba分词和情感词典的情感预测器。
    返回: 1 代表正面, 0 代表负面。
    """
    score = 0
    # 确保输入是字符串
    if not isinstance(text, str):
        return 0  # 或者返回一个中性值

    words = jieba.lcut(text)
    for word in words:
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
    return 1 if score >= 0 else 0


def run_word_segment_experiment():
    """
    执行分词优化策略实验的完整流程。
    """
    print("\n" + "=" * 20)
    print("--- 开始执行【实验二：分词策略优化】 ---")

    # 1. 加载自定义词典
    # 我们使用最终由lexicon_formatter.py生成的、干净的词典
    user_dict_path = '../data/weibo_user_dict_curated.txt'
    if os.path.exists(user_dict_path):
        jieba.load_userdict(user_dict_path)
        print(f"成功加载自定义词典: {user_dict_path}")
    else:
        print(f"警告：未找到自定义词典文件: {user_dict_path}")
        print("将仅使用默认分词。建议先运行新词发现和格式化脚本。")

    # 2. 自动获取并加载SnowNLP情感词典
    pos_path, neg_path = get_snownlp_dicts()
    positive_words, negative_words = load_sentiment_words(pos_path, neg_path)
    print(f"加载了 {len(positive_words)} 个正面情感词和 {len(negative_words)} 个负面情感词。")

    # 3. 加载测试数据
    test_df = pd.read_csv('../data/weibo_test_data.csv')
    test_df = test_df.sample(n=1000, random_state=42)  # 同样在1000条样本上测试
    print(f"加载了 {len(test_df)} 条测试数据。")

    # 4. 进行预测
    print("正在使用自定义预测器进行情感预测...")
    test_df['predicted_label'] = test_df['cleaned_text'].apply(
        lambda text: custom_sentiment_predict(text, positive_words, negative_words)
    )
    print("预测完成。")

    # 5. 准备真实标签和预测标签
    y_true = test_df['label']
    y_pred = test_df['predicted_label']
    class_labels = ['负面 (Negative)', '正面 (Positive)']

    # 6. 调用评估工具
    evaluate_and_visualize(
        y_true,
        y_pred,
        labels=class_labels,
        save_path='../reports/figures/word_segment_opt_confusion_matrix.png'
    )

    print("--- 【实验二】执行完毕 ---")


if __name__ == '__main__':
    run_word_segment_experiment()

