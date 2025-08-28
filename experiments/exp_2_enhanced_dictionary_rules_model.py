# file: experiments/exp_2_enhanced_dictionary_rules_model.py

import sys
import os
import jieba
import re  # 引入正则表达式库用于分句

# 导入我们的框架
# 这确保了脚本可以找到 utils, config 等模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_evaluator import ModelEvaluator
from experiments.data_loader import load_and_split_data
from config import CLASS_LABELS, CLASS_NAMES, FIGURES_DIR, PROJECT_ROOT


def load_boson_sentiment_dictionary():
    """
    加载 BosonNLP 情感词典。
    该词典包含情感词及其对应的情感分数。
    """
    senti_dict = {}
    dict_path = PROJECT_ROOT / 'data'/'Bosonnlp' / '情感词典（带有情感评分，BosonNLP_sentiment_score）'/'BosonNLP_sentiment_score.txt'
    # --- 路径配置结束 ---

    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    word = parts[0]
                    score = float(parts[1])
                    senti_dict[word] = score
        print(f"Successfully loaded BosonNLP sentiment dictionary. Total words: {len(senti_dict)}")
    except FileNotFoundError:
        print(f"Error: Sentiment dictionary not found at the specified path: {dict_path}")
        print("Please double-check the 'dict_path' variable in the 'load_boson_sentiment_dictionary' function.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the sentiment dictionary: {e}")
        return None

    return senti_dict


def rules_based_predict_enhanced(text, senti_dict, negation_words, degree_adverbs, conjunction_words):
    """
    【进阶版】基于词典和复杂规则进行情感预测。
    - 规则1: 以句子为单位进行分析，避免跨句干扰。
    - 规则2: 实现否定词和程度副词的“作用域(scope)”，它们会影响后面遇到的第一个情感词。
    - 规则3: 考虑转折连词，对转折后的内容赋予更高权重。
    """
    total_score = 0

    # 1. 将文本切分为句子
    sentences = re.split(r'[。！？；!?;\n]', text)

    for sentence in sentences:
        if not sentence.strip():
            continue

        words = jieba.lcut(sentence)
        sentence_score = 0
        modifier = 1.0  # 初始修饰符权重为1

        # 2. 在句子内部处理规则
        for i, word in enumerate(words):
            if word in senti_dict:
                current_score = senti_dict[word] * modifier
                sentence_score += current_score
                modifier = 1.0  # 情感词被修饰后，重置修饰符

            elif word in negation_words:
                modifier *= -1

            elif word in degree_adverbs:
                modifier *= degree_adverbs[word]

        # 3. 处理转折逻辑
        if any(word in conjunction_words for word in words):
            total_score += sentence_score * 1.5  # 为转折句赋予更高的权重
        else:
            total_score += sentence_score

    # 根据最终总分判断情感类别
    if total_score > 0:
        return 1
    elif total_score < 0:
        return -1
    else:
        return 0


def run_experiment_2_enhanced():
    """
    执行实验二的优化版：基于 BosonNLP 词典与复杂规则的模型评估。
    """
    print("\n" + "=" * 50)
    print("Running Experiment 2 (Enhanced): BosonNLP Dictionary and Advanced Rules")
    print("=" * 50)

    # 1. 加载数据
    print("------------------------------")
    print("Loading data...")
    _, X_test, _, y_test = load_and_split_data()
    if X_test is None:
        return
    print("------------------------------\n")

    # 2. 准备词典和规则集
    senti_dict = load_boson_sentiment_dictionary()
    if senti_dict is None:
        return  # 如果词典加载失败，则终止实验

    negation_words = {'不', '没', '无', '非', '莫', '弗', '勿', '毋', '未', '否', '别', '不要'}
    degree_adverbs = {'很': 1.25, '太': 1.5, '非常': 1.75, '极其': 2.0, '有点': 0.8, '稍微': 0.6, '最': 2.0,
                      '十分': 1.6}
    conjunction_words = {'但是', '但', '然而', '不过', '可是'}

    # 3. 模型预测
    print("Making predictions with enhanced dictionary and rules...")
    y_pred = X_test.apply(lambda text: rules_based_predict_enhanced(
        text, senti_dict, negation_words, degree_adverbs, conjunction_words
    ))
    print("Prediction complete.")

    # 4. 评估并保存结果
    print("\nEvaluating model performance...")
    evaluator = ModelEvaluator(
        y_true=y_test,
        y_pred=y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES
    )

    evaluator.print_report()

    # 定义所有图表的保存路径
    cm_path = FIGURES_DIR / 'exp2_enhanced_confusion_matrix.png'
    report_path = FIGURES_DIR / 'exp2_enhanced_classification_report.png'
    bar_chart_path = FIGURES_DIR / 'exp2_enhanced_metrics_bar_chart.png'

    # 调用所有绘图方法，生成全部三种图表
    print("\nGenerating and saving visualizations...")
    evaluator.plot_confusion_matrix(save_path=cm_path, title='增强规则模型-混淆矩阵')
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

    model_name = "Dict_AdvancedRules"
    all_performance[model_name] = evaluator.get_macro_f1_score()

    with open(performance_file, 'w', encoding='utf-8') as f:
        json.dump(all_performance, f, ensure_ascii=False, indent=4)

    print(f"\nPerformance for {model_name} saved to {performance_file.name}.")
    print("Experiment 2 (Enhanced) finished. Figures saved to reports/figures/")


if __name__ == '__main__':
    run_experiment_2_enhanced()