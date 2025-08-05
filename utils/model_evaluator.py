# /utils/model_evaluator.py (终极版)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

# 【新增】获取项目的绝对根目录路径
# Path(__file__) 是当前文件 (model_evaluator.py) 的路径
# .resolve() 将其转为绝对路径
# .parent.parent 会向上跳两级 (从 utils -> sentiment_analysis_project/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 设置matplotlib正常显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def evaluate_and_report(y_true, y_pred, labels, target_names):
    """计算并打印分类报告和准确率。"""
    print("\n模型性能评估报告:")
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, digits=4))
    accuracy = accuracy_score(y_true, y_pred)
    print(f"整体准确率 (Overall Accuracy): {accuracy:.4f}\n")
    return accuracy


def plot_and_save_confusion_matrix(y_true, y_pred, labels, display_labels, relative_save_path):
    """
    绘制并【使用绝对路径】保存混淆矩阵图。
    """
    # 【修改】将相对路径转换为项目内的绝对路径
    absolute_save_path = PROJECT_ROOT / relative_save_path

    # 使用新的绝对路径来确保目录存在
    absolute_save_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_labels, yticklabels=display_labels)
    plt.title('混淆矩阵 (Confusion Matrix)')
    plt.xlabel('预测标签 (Predicted Label)')
    plt.ylabel('真实标签 (True Label)')

    # 【新增】打印出绝对路径，用于最终确认
    print(f"【绝对路径诊断】: 正在尝试将文件保存到: {absolute_save_path}")

    # 使用绝对路径进行保存
    plt.savefig(absolute_save_path)
    print(f"混淆矩阵图已保存至: {absolute_save_path}")
    plt.close()