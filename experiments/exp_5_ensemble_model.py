# experiments/exp_5_ensemble_model.py
import pandas as pd
import jieba
import sys
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.append('..')
from utils.model_evaluator import evaluate_and_visualize
# 我们需要复用实验三的逻辑，所以直接从exp_3导入
from exp_3_rules_opt import load_sentiment_dicts, advanced_sentiment_predict


def run_ensemble_experiment():
    """
    执行基于Stacking的模型集成实验。
    """
    print("\n" + "=" * 20)
    print("--- 开始执行【实验五：模型集成】 ---")

    # --- Step 1: 加载所有需要的数据 ---
    print("Step 1: 正在加载所有数据集...")
    # 大规模训练集，用于训练基础的ML模型
    ml_train_df = pd.read_csv('../data/online_shopping_10_cats.csv').dropna(subset=['review', 'label'])
    # 微博训练集，用于训练我们的“掌门人”（元模型）
    meta_train_df = pd.read_csv('../data/weibo_train_data.csv').dropna(subset=['cleaned_text', 'label'])
    # 微博测试集，用于最终的“大比武”
    test_df = pd.read_csv('../data/weibo_test_data.csv').dropna(subset=['cleaned_text', 'label'])
    print("所有数据集加载完毕。")

    # --- Step 2: 准备我们的两位“高手”（基础模型） ---
    print("\nStep 2: 正在准备基础模型...")
    # 准备自定义分词词典
    user_dict_path = '../data/weibo_user_dict_curated.txt'
    if os.path.exists(user_dict_path):
        jieba.load_userdict(user_dict_path)

    # “高手A”：实验四的机器学习模型
    def jieba_tokenizer(text):
        return jieba.lcut(text)

    ml_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None)),
        ('classifier', MultinomialNB())
    ])
    print("  - 高手A (机器学习模型) 已准备就绪。")

    # “高手B”：实验三的规则模型
    sentiment_dictionary = load_sentiment_dicts()
    negation_words = {'不', '没', '无', '非', '莫', '弗', '勿', '毋', '未', '否', '别'}
    degree_adverbs = {'很': 1.2, '太': 1.4, '非常': 1.6, '极其': 2.0, '有点': 0.8, '稍微': 0.7}
    print("  - 高手B (规则模型) 已准备就绪。")

    # --- Step 3: 训练“高手A” ---
    print("\nStep 3: 正在大规模数据集上训练机器学习模型...")
    ml_pipeline.fit(ml_train_df['review'].astype(str), ml_train_df['label'])
    print("机器学习模型训练完成。")

    # --- Step 4: 训练“掌门人”（元模型） ---
    print("\nStep 4: 正在生成特征并训练元模型...")
    # 1. 让两位高手对“微博训练集”进行预测，生成新的特征
    meta_X_train = pd.DataFrame()
    meta_X_train['ml_pred'] = ml_pipeline.predict(meta_train_df['cleaned_text'])
    meta_X_train['rule_pred'] = meta_train_df['cleaned_text'].apply(
        lambda text: advanced_sentiment_predict(text, sentiment_dictionary, negation_words, degree_adverbs)
    )
    meta_y_train = meta_train_df['label']

    # 2. 训练元模型（逻辑回归）
    meta_model = LogisticRegression()
    meta_model.fit(meta_X_train, meta_y_train)
    print("元模型训练完成。")

    # --- Step 5: 最终决战！在测试集上进行集成预测 ---
    print("\nStep 5: 正在微博测试集上进行最终集成预测...")
    # 1. 同样，先让两位高手对“微博测试集”进行预测
    meta_X_test = pd.DataFrame()
    meta_X_test['ml_pred'] = ml_pipeline.predict(test_df['cleaned_text'])
    meta_X_test['rule_pred'] = test_df['cleaned_text'].apply(
        lambda text: advanced_sentiment_predict(text, sentiment_dictionary, negation_words, degree_adverbs)
    )

    # 2. 让“掌门人”根据两位高手的意见，做出最终的裁决
    final_predictions = meta_model.predict(meta_X_test)

    # --- Step 6: 评估最终模型的性能 ---
    print("\nStep 6: 评估最终集成模型的性能...")
    y_true = test_df['label']
    class_labels = ['负面 (Negative)', '正面 (Positive)']
    evaluate_and_visualize(
        y_true,
        final_predictions,
        labels=class_labels,
        save_path='../reports/figures/ensemble_model_confusion_matrix.png'
    )

    print("--- 【实验五】执行完毕 ---")


if __name__ == '__main__':
    run_ensemble_experiment()
