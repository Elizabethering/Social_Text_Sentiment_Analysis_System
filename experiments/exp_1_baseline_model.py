import sys
import os
from snownlp import SnowNLP
import pandas as pd # 引入pandas

# 导入我们创建的框架
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_evaluator import ModelEvaluator
from experiments.data_loader import load_and_split_data
from config import CLASS_LABELS, CLASS_NAMES, FIGURES_DIR, PROJECT_ROOT


def snownlp_predict(text):
    """自定义SnowNLP的预测逻辑，将其映射到-1, 0, 1"""
    # 【优化】增加对非字符串输入处理，避免潜在错误
    if not isinstance(text, str):
        return 0 # 对于空值或非字符串，返回中性
    try:
        score = SnowNLP(text).sentiments
        if score > 0.6:
            return 1
        elif score < 0.4:
            return -1
        else:
            return 0
    except Exception:
        return 0 # 如果SnowNLP处理失败，同样返回中性


def run_experiment_1():
    """
    执行实验一：SnowNLP基线模型评估。
    """
    print("\n" + "=" * 50)
    print("Running Experiment 1: SnowNLP Baseline Model")
    print("=" * 50)

    # 1. 加载数据
    _, X_test, _, y_test = load_and_split_data()
    if X_test is None:
        return

    # 2. 【核心修改】对大数据集进行抽样
    # 由于测试集非常大，直接在上面运行SnowNLP会非常耗时。
    # 我们随机抽取一个子集（例如5000个样本）来进行快速评估。
    print(f"Original test set size: {len(X_test)}. This is too large for SnowNLP.")
    sample_size = 5000
    if len(X_test) > sample_size:
        print(f"Taking a random sample of {sample_size} for evaluation...")
        X_test_sample = X_test.sample(n=sample_size, random_state=42)
        y_test_sample = y_test.loc[X_test_sample.index]
    else:
        X_test_sample = X_test
        y_test_sample = y_test

    # 3. 模型预测 (在抽样后的数据上进行)
    print("Making predictions with SnowNLP on the sample data...")
    y_pred = X_test_sample.apply(snownlp_predict)

    # 4. 评估并保存结果
    print("Evaluating model performance...")
    evaluator = ModelEvaluator(
        y_true=y_test_sample, # 注意：这里使用抽样后的真实标签
        y_pred=y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES
    )

    evaluator.print_report()

    # 定义图表保存路径
    cm_path = FIGURES_DIR / 'exp1_snownlp_confusion_matrix.png'
    report_path = FIGURES_DIR / 'exp1_snownlp_classification_report.png'
    bar_chart_path = FIGURES_DIR / 'exp1_snownlp_metrics_bar_chart.png'

    # 调用所有绘图方法
    evaluator.plot_confusion_matrix(save_path=cm_path)
    evaluator.plot_classification_report(save_path=report_path)
    evaluator.plot_metrics_bar_chart(save_path=bar_chart_path)

    # 5. 保存性能指标用于最终对比
    import json
    performance_file = PROJECT_ROOT / 'all_models_performance.json'
    if performance_file.exists():
        with open(performance_file, 'r', encoding='utf-8') as f:
            all_performance = json.load(f)
    else:
        all_performance = {}

    model_name = "SnowNLP_Baseline"
    all_performance[model_name] = evaluator.get_macro_f1_score()

    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(all_performance, f, ensure_ascii=False, indent=4)

    print(f"\nPerformance for {model_name} saved.")
    print("Experiment 1 finished. Figures saved to reports/figures/")


if __name__ == '__main__':
    run_experiment_1()
