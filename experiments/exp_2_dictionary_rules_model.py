import sys
import os
import jieba

# 导入我们的框架
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_evaluator import ModelEvaluator
from experiments.data_loader import load_and_split_data
from config import CLASS_LABELS, CLASS_NAMES, FIGURES_DIR, PROJECT_ROOT


def load_sentiment_dictionary():
    """
    加载情感词典。
    这里我们简化流程，直接加载SnowNLP自带的词典作为示例。
    """
    senti_dict = {}
    try:
        import snownlp
        package_path = os.path.dirname(snownlp.__file__)
        pos_path = os.path.join(package_path, 'sentiment', 'pos.txt')
        neg_path = os.path.join(package_path, 'sentiment', 'neg.txt')

        with open(pos_path, 'r', encoding='utf-8') as f:
            for word in f:
                senti_dict[word.strip()] = 1.0
        with open(neg_path, 'r', encoding='utf-8') as f:
            for word in f:
                senti_dict[word.strip()] = -1.0
        print("Successfully loaded sentiment dictionary from SnowNLP.")
    except Exception as e:
        print(f"Error loading sentiment dictionary: {e}")
    return senti_dict


def rules_based_predict(text, senti_dict, negation_words, degree_adverbs):
    """
    基于词典和规则进行情感预测。
    """
    score = 0
    words = jieba.lcut(text)

    for i, word in enumerate(words):
        if word in senti_dict:
            senti_score = senti_dict[word]
            modifier = 1
            if i > 0:
                prev_word = words[i - 1]
                if prev_word in negation_words:
                    modifier *= -1
                elif prev_word in degree_adverbs:
                    modifier *= degree_adverbs[prev_word]
            score += senti_score * modifier

    if score > 0:
        return 1
    elif score < 0:
        return -1
    else:
        return 0


def run_experiment_2():
    """
    执行实验二：基于词典与规则的优化模型评估。
    """
    print("\n" + "=" * 50)
    print("Running Experiment 2: Dictionary and Rules-based Model")
    print("=" * 50)

    # 1. 加载数据
    _, X_test, _, y_test = load_and_split_data()
    if X_test is None:
        return

    # 2. 准备词典和规则
    senti_dict = load_sentiment_dictionary()
    negation_words = {'不', '没', '无', '非', '莫', '弗', '勿', '毋', '未', '否', '别', '不要'}
    degree_adverbs = {'很': 1.25, '太': 1.5, '非常': 1.75, '极其': 2.0, '有点': 0.8, '稍微': 0.6, '最': 2.0}

    # 3. 模型预测
    print("Making predictions with dictionary and rules...")
    y_pred = X_test.apply(lambda text: rules_based_predict(text, senti_dict, negation_words, degree_adverbs))

    # 4. 评估并保存结果
    print("Evaluating model performance...")
    evaluator = ModelEvaluator(
        y_true=y_test,
        y_pred=y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES
    )

    evaluator.print_report()

    # 定义图表保存路径
    cm_path = FIGURES_DIR / 'exp2_dict_rules_confusion_matrix.png'
    report_path = FIGURES_DIR / 'exp2_dict_rules_classification_report.png'
    bar_chart_path = FIGURES_DIR / 'exp2_dict_rules_metrics_bar_chart.png' # 【新增】条形图路径

    # 调用所有绘图方法
    evaluator.plot_confusion_matrix(save_path=cm_path)
    evaluator.plot_classification_report(save_path=report_path)
    evaluator.plot_metrics_bar_chart(save_path=bar_chart_path) # 【新增】调用新方法

    # 5. 保存性能指标用于最终对比
    import json
    performance_file = PROJECT_ROOT / 'all_models_performance.json'
    if performance_file.exists():
        with open(performance_file, 'r', encoding='utf-8') as f:
            all_performance = json.load(f)
    else:
        all_performance = {}

    model_name = "Dictionary_Rules"
    all_performance[model_name] = evaluator.get_macro_f1_score()

    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(all_performance, f, ensure_ascii=False, indent=4)

    print(f"\nPerformance for {model_name} saved.")
    print("Experiment 2 finished. Figures saved to reports/figures/")


if __name__ == '__main__':
    run_experiment_2()

