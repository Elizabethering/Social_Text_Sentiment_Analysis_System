# experiments/exp_3_advanced_rules_opt.py
import pandas as pd
import jieba
import sys
import os

# --- GPS导航系统：动态定位项目根目录 ---
# 获取当前脚本(exp_3_advanced_rules_opt.py)的绝对路径
# __file__ 是一个内置变量，代表当前文件的路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 从脚本目录(experiments)向上走一级，就得到了项目的根目录
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 将项目根目录添加到系统路径中，这样才能导入utils模块
sys.path.append(PROJECT_ROOT)
from utils.model_evaluator import evaluate_and_visualize


# 创建一个辅助函数，用于拼接项目根目录和相对路径
def get_project_path(relative_path):
    """根据项目根目录返回一个文件的绝对路径"""
    return os.path.join(PROJECT_ROOT, relative_path)


# --- GPS导航系统结束 ---


def load_advanced_sentiment_dict():
    """
    加载大规模情感词典(BosonNLP)并融合我们自己的领域词典。
    """
    senti_dict = {}

    # 1. 加载BosonNLP大规模情感词典 (使用GPS函数定位)
    # --- 关键修正：更新路径以匹配你的文件夹结构 ---
    boson_dict_path = get_project_path(
        'data/BosonNLP/情感词典（带有情感评分，BosonNLP_sentiment_score）/BosonNLP_sentiment_score.txt')
    if os.path.exists(boson_dict_path):
        with open(boson_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) == 2:
                    senti_dict[parts[0]] = float(parts[1])
        print(f"成功加载大规模情感词典: {boson_dict_path}")
    else:
        print(f"警告: 未找到大规模情感词典 {boson_dict_path}，模型性能会受影响。")

    # 2. 加载并融合我们自己的领域情感词典 (使用GPS函数定位)
    domain_dict_path = get_project_path('data/weibo_domain_sentiment_dict.txt')
    if os.path.exists(domain_dict_path):
        with open(domain_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) == 2:
                    senti_dict[parts[0]] = float(parts[1])
        print(f"成功加载并融合领域情感词典: {domain_dict_path}")

    return senti_dict


def advanced_sentiment_predict(text, senti_dict, negation_words, degree_adverbs):
    """
    一个更高级的情感预测器，考虑了否定词和程度副词。
    """
    score = 0
    sentiment_stack = []

    if not isinstance(text, str):
        return 0

    words = jieba.lcut(text)
    for i, word in enumerate(words):
        weight = senti_dict.get(word, 0)
        if word in negation_words:
            sentiment_stack.append('negation')
        elif word in degree_adverbs:
            sentiment_stack.append(degree_adverbs[word])
        elif weight != 0:
            modifier = 1
            while sentiment_stack:
                mod = sentiment_stack.pop()
                if mod == 'negation':
                    modifier *= -1
                else:
                    modifier *= mod
            score += weight * modifier

    return 1 if score >= 0 else 0


def run_advanced_rules_experiment():
    """执行高级规则优化实验"""
    print("\n" + "=" * 20)
    print("--- 开始执行【实验三：高级规则优化】 ---")

    # 1. 加载分词词典 (使用GPS函数定位)
    user_dict_path = get_project_path('data/weibo_user_dict_curated.txt')
    if os.path.exists(user_dict_path):
        jieba.load_userdict(user_dict_path)
        print(f"成功加载自定义分词词典: {user_dict_path}")

    # 2. 加载增强版情感词典
    sentiment_dictionary = load_advanced_sentiment_dict()
    print(f"总计加载了 {len(sentiment_dictionary)} 个带权重的情感词。")

    # 3. 定义规则词
    negation_words = {'不', '没', '无', '非', '莫', '弗', '勿', '毋', '未', '否', '别', '不要'}
    degree_adverbs = {'很': 1.25, '太': 1.5, '非常': 1.75, '极其': 2.0, '有点': 0.8, '稍微': 0.6, '最': 2.0}

    # 4. 加载测试数据 (使用GPS函数定位)
    test_df = pd.read_csv(get_project_path('data/weibo_test_data.csv'))
    # 这次我们在完整的测试集上运行，看看最终效果！
    # test_df = test_df.sample(n=1000, random_state=42)
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
        # 使用GPS函数定位
        save_path=get_project_path('reports/figures/advanced_rules_opt_confusion_matrix.png')
    )

    print("--- 【实验三 高级版】执行完毕 ---")


if __name__ == '__main__':
    run_advanced_rules_experiment()
