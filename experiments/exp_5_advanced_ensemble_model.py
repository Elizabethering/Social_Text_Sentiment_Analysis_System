# experiments/exp_5_ensemble_model.py (Advanced Version)
import pandas as pd
import jieba
import sys
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm import tqdm

sys.path.append('..')
from utils.model_evaluator import evaluate_and_visualize
from exp_3_advanced_rules_opt import load_advanced_sentiment_dict, advanced_sentiment_predict


# 创建一个新函数，使其返回原始分数而不是0/1
def get_sentiment_score(text, senti_dict, negation_words, degree_adverbs):
    """
    计算情感原始分数，而不是直接返回0或1。
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
    return score


def run_advanced_ensemble_experiment():
    """
    执行基于Stacking和置信度分数的终极模型集成实验。
    """
    print("\n" + "=" * 20)
    print("--- 开始执行【实验五：终极集成模型】 ---")

    # --- Step 1: 加载数据 ---
    print("Step 1: 正在加载所有数据集...")
    ml_train_df = pd.read_csv('../data/online_shopping_10_cats.csv').dropna(subset=['review', 'label'])
    meta_train_df = pd.read_csv('../data/weibo_train_data.csv').dropna(subset=['cleaned_text', 'label'])
    test_df = pd.read_csv('../data/weibo_test_data.csv').dropna(subset=['cleaned_text', 'label'])
    print("所有数据集加载完毕。")

    # --- Step 2: 准备基础模型 ---
    print("\nStep 2: 正在准备基础模型...")
    user_dict_path = '../data/weibo_user_dict_curated.txt'
    if os.path.exists(user_dict_path):
        jieba.load_userdict(user_dict_path)

    # “高手A”：支持向量机 (SVM) 模型
    def jieba_tokenizer(text):
        return jieba.lcut(text)

    # LinearSVC本身不直接支持概率预测，我们用CalibratedClassifierCV来包装它
    svm_model = LinearSVC(random_state=42, dual=True)
    calibrated_svm = CalibratedClassifierCV(svm_model, method='sigmoid', cv=3)
    ml_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None)),
        ('classifier', calibrated_svm)
    ])
    print("  - 高手A (支持向量机模型) 已准备就绪。")

    # “高手B”：高级规则模型
    sentiment_dictionary = load_advanced_sentiment_dict()
    negation_words = {'不', '没', '无', '非', '莫', '弗', '勿', '毋', '未', '否', '别', '不要'}
    degree_adverbs = {'很': 1.25, '太': 1.5, '非常': 1.75, '极其': 2.0, '有点': 0.8, '稍微': 0.6, '最': 2.0}
    print("  - 高手B (高级规则模型) 已准备就绪。")

    # --- Step 3: 训练“高手A” ---
    print("\nStep 3: 正在大规模数据集上训练SVM模型...")
    ml_pipeline.fit(ml_train_df['review'].astype(str), ml_train_df['label'])
    print("SVM模型训练完成。")

    # --- Step 4: 训练“掌门人”（元模型） ---
    print("\nStep 4: 正在生成高级特征并训练元模型...")
    # 1. 生成新的特征：这次是概率和分数！
    meta_X_train = pd.DataFrame()
    # 获取SVM模型的预测概率
    ml_train_probs = ml_pipeline.predict_proba(meta_train_df['cleaned_text'])
    meta_X_train['ml_prob_neg'] = ml_train_probs[:, 0]
    meta_X_train['ml_prob_pos'] = ml_train_probs[:, 1]
    # 获取规则模型的原始分数
    meta_X_train['rule_score'] = meta_train_df['cleaned_text'].apply(
        lambda text: get_sentiment_score(text, sentiment_dictionary, negation_words, degree_adverbs)
    )
    meta_y_train = meta_train_df['label']

    # 2. 训练元模型
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(meta_X_train, meta_y_train)
    print("元模型训练完成。")

    # --- Step 5: 最终决战！ ---
    print("\nStep 5: 正在微博测试集上进行最终集成预测...")
    # 1. 同样，生成测试集的高级特征
    meta_X_test = pd.DataFrame()
    ml_test_probs = ml_pipeline.predict_proba(test_df['cleaned_text'])
    meta_X_test['ml_prob_neg'] = ml_test_probs[:, 0]
    meta_X_test['ml_prob_pos'] = ml_test_probs[:, 1]
    meta_X_test['rule_score'] = test_df['cleaned_text'].apply(
        lambda text: get_sentiment_score(text, sentiment_dictionary, negation_words, degree_adverbs)
    )

    # 2. 让“掌门人”做出最终裁决
    final_predictions = meta_model.predict(meta_X_test)

    # --- Step 6: 评估最终模型的性能 ---
    print("\nStep 6: 评估最终集成模型的性能...")
    y_true = test_df['label']
    class_labels = ['负面 (Negative)', '正面 (Positive)']
    evaluate_and_visualize(
        y_true,
        final_predictions,
        labels=class_labels,
        save_path='../reports/figures/advanced_ensemble_model_confusion_matrix.png'
    )

    print("--- 【实验五 威力加强版】执行完毕 ---")


if __name__ == '__main__':
    run_advanced_ensemble_experiment()
