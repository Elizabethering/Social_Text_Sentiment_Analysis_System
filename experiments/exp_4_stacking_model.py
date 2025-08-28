import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
import jieba
import json

# 导入我们的框架
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_evaluator import ModelEvaluator
from experiments.data_loader import load_and_split_data
from config import CLASS_LABELS, CLASS_NAMES, FIGURES_DIR, PROJECT_ROOT


def run_experiment_4_updated():
    """
    执行实验四（更新版）：基于SVM和LightGBM的Stacking集成学习模型评估。
    """
    print("\n" + "=" * 50)
    print("Running Experiment 4 (Updated): Stacking Ensemble with SVM and LightGBM")
    print("=" * 50)

    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_and_split_data()
    if X_train is None:
        return

    # 【注意】由于SVM在基模型中训练耗时较长，
    # 如果在完整数据集上运行过慢，可以考虑对X_train和y_train进行抽样
    train_sample_size = 50000
    X_train = X_train.sample(n=train_sample_size, random_state=42)
    y_train = y_train.loc[X_train.index]
    print(f"Using a sample of {train_sample_size} for training due to SVM complexity.")


    # 2. 【核心修改】定义基础模型
    # 根据实验三的结果，我们选择表现最好的SVM和LightGBM作为基模型
    estimators = [
        ('svm', Pipeline([
            ('tfidf_svm', TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)))),
            ('clf_svm', SVC(kernel='linear', probability=True, random_state=42)) # probability=True是Stacking所必需的
        ])),
        ('lgbm', Pipeline([
            ('tfidf_lgbm', TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)))),
            ('clf_lgbm', LGBMClassifier(n_estimators=200, objective='multiclass', random_state=42))
        ]))
    ]

    # 3. 定义元模型（次级模型）
    # 通常选用一个简单的线性模型，如逻辑回归
    final_estimator = LogisticRegression(max_iter=1000)

    # 4. 创建Stacking分类器
    # cv=3 表示在训练基模型时使用3折交叉验证，这有助于增强模型的泛化能力
    stacking_classifier = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=3,
        passthrough=False # passthrough=False意味着只有基模型的预测结果会作为元模型的输入
    )

    # 5. 训练和预测
    print("Training and predicting with Stacking Classifier (SVM + LightGBM)...")
    stacking_classifier.fit(X_train, y_train)
    y_pred = stacking_classifier.predict(X_test)

    # 6. 评估和保存结果
    print("Evaluating Stacking Classifier...")
    evaluator = ModelEvaluator(y_test, y_pred, labels=CLASS_LABELS, target_names=CLASS_NAMES)
    evaluator.print_report()

    # 定义图表保存路径
    cm_path = FIGURES_DIR / 'exp4_stacking_updated_confusion_matrix.png'
    report_path = FIGURES_DIR / 'exp4_stacking_updated_classification_report.png'
    bar_chart_path = FIGURES_DIR / 'exp4_stacking_updated_metrics_bar_chart.png'

    # 调用所有绘图方法
    evaluator.plot_confusion_matrix(save_path=cm_path)
    evaluator.plot_classification_report(save_path=report_path)
    evaluator.plot_metrics_bar_chart(save_path=bar_chart_path)

    # 7. 保存性能指标
    performance_file = PROJECT_ROOT / 'all_models_performance.json'
    if performance_file.exists():
        with open(performance_file, 'r', encoding='utf-8') as f:
            all_performance = json.load(f)
    else:
        all_performance = {}

    model_name = "Stacking_Ensemble_Updated"
    all_performance[model_name] = evaluator.get_macro_f1_score()

    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(all_performance, f, ensure_ascii=False, indent=4)

    print(f"\nPerformance for {model_name} saved.")
    print("Experiment 4 (Updated) finished. Figures saved to reports/figures/")


if __name__ == '__main__':
    run_experiment_4_updated()