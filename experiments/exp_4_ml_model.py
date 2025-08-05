# experiments/exp_4_ml_model.py
import pandas as pd
import jieba
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

sys.path.append('..')
from utils.model_evaluator import evaluate_and_visualize


def run_ml_experiment():
    """
    执行基于机器学习的核心模型替换实验。
    """
    print("\n" + "=" * 20)
    print("--- 开始执行【实验四：核心模型替换】 ---")

    # --- Step 1: 加载并准备训练数据 (我们的“大型训练基地”) ---
    print("Step 1: 正在加载大规模电商评论数据用于训练...")
    train_corpus_path = '../data/online_shopping_10_cats.csv'
    try:
        train_df = pd.read_csv(train_corpus_path)
        # 数据集有两列: 'cat' 和 'review'，以及标签 'label' (1=正面, 0=负面)
        train_df.dropna(subset=['review', 'label'], inplace=True)
        X_train_text = train_df['review'].astype(str)
        y_train = train_df['label']
        print(f"成功加载 {len(train_df)} 条训练数据。")
    except FileNotFoundError:
        print(f"错误：找不到训练数据 '{train_corpus_path}'。请先下载。")
        return

    # --- Step 2: 加载测试数据 (我们的“官方比武场”) ---
    print("\nStep 2: 正在加载微博测试数据...")
    test_df = pd.read_csv('../data/weibo_test_data.csv')
    X_test_text = test_df['cleaned_text'].astype(str)
    y_test = test_df['label']
    print(f"成功加载 {len(test_df)} 条测试数据。")

    # --- Step 3: 构建机器学习流水线 (Pipeline) ---
    print("\nStep 3: 正在构建机器学习流水线...")
    # 加载我们的自定义分词词典，让模型也能认识网络用语
    user_dict_path = '../data/weibo_user_dict_curated.txt'
    if os.path.exists(user_dict_path):
        jieba.load_userdict(user_dict_path)
        print("流水线已配置自定义分词词典。")

    # 定义一个分词函数，供TfidfVectorizer使用
    def jieba_tokenizer(text):
        return jieba.lcut(text)

    # Pipeline能将多个步骤串联起来，像流水线一样工作
    # 步骤1: 'tfidf' - 将文本转换为TF-IDF向量
    # 步骤2: 'classifier' - 使用朴素贝叶斯分类器进行分类
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=jieba_tokenizer, token_pattern=None)),
        ('classifier', MultinomialNB())
    ])
    print("流水线构建完成: Jieba分词 -> TF-IDF向量化 -> 朴素贝叶斯分类器")

    # --- Step 4: 训练模型 ---
    print("\nStep 4: 正在训练模型，这可能需要一两分钟...")
    pipeline.fit(X_train_text, y_train)
    print("模型训练完成！")

    # --- Step 5: 在测试集上进行预测和评估 ---
    print("\nStep 5: 正在微博测试集上进行预测和评估...")
    y_pred = pipeline.predict(X_test_text)

    class_labels = ['负面 (Negative)', '正面 (Positive)']
    evaluate_and_visualize(
        y_test,
        y_pred,
        labels=class_labels,
        save_path='../reports/figures/ml_model_confusion_matrix.png'
    )

    print("--- 【实验四】执行完毕 ---")


if __name__ == '__main__':
    run_ml_experiment()
