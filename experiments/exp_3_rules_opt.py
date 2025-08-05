# experiments/exp_3_rules_opt.py
import pandas as pd
import jieba
import sys
import os

sys.path.append('..')
from utils.model_evaluator import evaluate_and_visualize


def get_snownlp_dicts():
    """自动查找并返回SnowNLP自带的情感词典路径。"""
    try:
        import snownlp
        package_path = os.path.dirname(snownlp.__file__)
        pos_path = os.path.join(package_path, 'sentiment', 'pos.txt')
        neg_path = os.path.join(package_path, 'sentiment', 'neg.txt')
        if not os.path.exists(pos_path) or not os.path.exists(neg_path):
            raise FileNotFoundError
        return pos_path, neg_path
    except (ImportError, FileNotFoundError):
        print("警告：无法从SnowNLP自动加载情感词典。程序将退出。")
        sys.exit(1)


def load_sentiment_dicts():
    """
    加载所有情感词典：基础词典 + 领域词典。
    返回一个合并后的情感词典，格式为 {'词语': 权重}。
    """
    senti_dict = {}

    # 1. 加载SnowNLP基础词典
    pos_path, neg_path = get_snownlp_dicts()
    with open(pos_path, 'r', encoding='utf-8') as f:
        for word in f:
            senti_dict[word.strip()] = 1.0  # 基础正面词权重为1
    with open(neg_path, 'r', encoding='utf-8') as f:
        for word in f:
            senti_dict[word.strip()] = -1.0  # 基础负面词权重为-1

    # 2. 加载并融合我们自己的领域情感词典
    domain_dict_path = '../data/weibo_domain_sentiment_dict.txt'
    if os.path.exists(domain_dict_path):
        with open(domain_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) == 2:
                    # 如果领域词典中的词已存在，则用领域权重覆盖基础权重
                    senti_dict[parts[0]] = float(parts[1])
        print(f"成功加载并融合领域情感词典: {domain_dict_path}")

    return senti_dict


def advanced_sentiment_predict(text, senti_dict, negation_words, degree_adverbs):
    """
    一个更高级的情感预测器，考虑了否定词和程度副词。
    返回: 1 代表正面, 0 代表负面。
    """
    score = 0
    sentiment_stack = []  # 用于处理否定和程度副词的栈

    if not isinstance(text, str):
        return 0

    words = jieba.lcut(text)
    for i, word in enumerate(words):
        # 1. 获取当前词的情感权重
        weight = senti_dict.get(word, 0)

        # 2. 处理否定词和程度副词
        if word in negation_words:
            sentiment_stack.append('negation')
        elif word in degree_adverbs:
            sentiment_stack.append(degree_adverbs[word])
        # 如果遇到情感词
        elif weight != 0:
            # 查看前面是否有否定或程度副词
            modifier = 1
            while sentiment_stack:
                mod = sentiment_stack.pop()
                if mod == 'negation':
                    modifier *= -1
                else:  # 是程度副词
                    modifier *= mod

            score += weight * modifier

    return 1 if score >= 0 else 0


def run_rules_experiment():
    """执行情感词典与规则优化实验"""
    print("\n" + "=" * 20)
    print("--- 开始执行【实验三：情感词典与规则优化】 ---")

    # 1. 加载分词词典
    user_dict_path = '../data/weibo_user_dict_curated.txt'
    if os.path.exists(user_dict_path):
        jieba.load_userdict(user_dict_path)
        print(f"成功加载自定义分词词典: {user_dict_path}")

    # 2. 加载情感词典
    sentiment_dictionary = load_sentiment_dicts()
    print(f"总计加载了 {len(sentiment_dictionary)} 个情感词。")

    # 3. 定义规则词
    negation_words = {'不', '没', '无', '非', '莫', '弗', '勿', '毋', '未', '否', '别'}
    degree_adverbs = {'很': 1.2, '太': 1.4, '非常': 1.6, '极其': 2.0, '有点': 0.8, '稍微': 0.7}

    # 4. 加载测试数据
    test_df = pd.read_csv('../data/weibo_test_data.csv')
    test_df = test_df.sample(n=1000, random_state=42)
    print(f"加载了 {len(test_df)} 条测试数据。")

    # 5. 进行预测
    print("正在使用高级规则预测器进行情感预测...")
    test_df['predicted_label'] = test_df['cleaned_text'].apply(
        lambda text: advanced_sentiment_predict(text, sentiment_dictionary, negation_words, degree_adverbs)
    )
    print("预测完成。")

    # 6. 评估
    y_true = test_df['label']
    y_pred = test_df['predicted_label']
    class_labels = ['负面 (Negative)', '正面 (Positive)']
    evaluate_and_visualize(
        y_true,
        y_pred,
        labels=class_labels,
        save_path='../reports/figures/rules_opt_confusion_matrix.png'
    )

    print("--- 【实验三】执行完毕 ---")


if __name__ == '__main__':
    run_rules_experiment()
