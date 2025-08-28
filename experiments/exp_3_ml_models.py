import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
import jieba
import json

# 导入我们的框架
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_evaluator import ModelEvaluator
from experiments.data_loader import load_and_split_data
from config import CLASS_LABELS, CLASS_NAMES, FIGURES_DIR, PROJECT_ROOT


def run_experiment_3():
    """
    执行实验三：横向对比多种传统机器学习模型。
    """
    print("\n" + "=" * 50)
    print("Running Experiment 3: Traditional ML Models Comparison")
    print("=" * 50)

    # 1. 加载完整数据集
    X_train, X_test, y_train, y_test = load_and_split_data()
    if X_train is None:
        return

    # 2. 【核心修改】仅为SVM创建专用的抽样数据集
    # 因为SVM在大型数据集上训练非常缓慢，我们为其准备一个较小的样本。
    svm_train_sample_size = 30000  # 用于SVM训练的样本数
    svm_test_sample_size = 10000  # 用于SVM测试的样本数

    print(f"Original training set size: {len(X_train)}")
    print(f"Original test set size: {len(X_test)}")
    print(
        f"A smaller sample of {svm_train_sample_size} training and {svm_test_sample_size} test points will be used for SVM.")

    X_train_svm_sample = X_train.sample(n=svm_train_sample_size, random_state=42)
    y_train_svm_sample = y_train.loc[X_train_svm_sample.index]

    X_test_svm_sample = X_test.sample(n=svm_test_sample_size, random_state=42)
    y_test_svm_sample = y_test.loc[X_test_svm_sample.index]

    # 3. 定义多个机器学习模型
    models_to_evaluate = {
        "Naive_Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)))),
            ('clf', MultinomialNB())
        ]),
        "SVM": Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)))),
            ('clf', SVC(kernel='linear', probability=True, random_state=42))
        ]),
        "LightGBM": Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)))),
            ('clf', LGBMClassifier(n_estimators=200, objective='multiclass', random_state=42))
        ])
    }

    # 初始化性能记录字典
    performance_file = PROJECT_ROOT / 'all_models_performance.json'
    if performance_file.exists():
        with open(performance_file, 'r', encoding='utf-8') as f:
            all_performance = json.load(f)
    else:
        all_performance = {}

    # 4. 循环训练、评估并保存每个模型的结果
    for model_name, pipeline in models_to_evaluate.items():
        print("\n" + "-" * 20 + f" Training and evaluating: {model_name} " + "-" * 20)

        # 【核心修改】根据模型名称选择使用的数据集
        if model_name == "SVM":
            print("--> Using smaller sampled dataset for SVM...")
            pipeline.fit(X_train_svm_sample, y_train_svm_sample)
            y_pred = pipeline.predict(X_test_svm_sample)
            evaluator = ModelEvaluator(y_test_svm_sample, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES)
        else:
            print("--> Using full dataset...")
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            evaluator = ModelEvaluator(y_test, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES)

        evaluator.print_report()

        # 定义图表保存路径
        cm_path = FIGURES_DIR / f'exp3_{model_name}_confusion_matrix.png'
        report_path = FIGURES_DIR / f'exp3_{model_name}_classification_report.png'
        bar_chart_path = FIGURES_DIR / f'exp3_{model_name}_metrics_bar_chart.png'

        # 调用所有绘图方法
        evaluator.plot_confusion_matrix(save_path=cm_path)
        evaluator.plot_classification_report(save_path=report_path)
        evaluator.plot_metrics_bar_chart(save_path=bar_chart_path)

        # 记录模型性能
        all_performance[model_name] = evaluator.get_macro_f1_score()
        print(f"\nPerformance for {model_name} recorded.")

    # 5. 将所有模型的性能一次性写入文件
    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(all_performance, f, ensure_ascii=False, indent=4)
    print(f"\nAll model performances saved to {performance_file.name}.")

    print("\nExperiment 3 finished. All model figures saved to reports/figures/")


if __name__ == '__main__':
    run_experiment_3()
