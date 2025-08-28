# file: utils/model_evaluator.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np  # 引入numpy用于条形图绘制
import os

# 设置matplotlib正常显示中文，并使用'Agg'后端避免在无界面的服务器上出错
import matplotlib

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ModelEvaluator:
    """
    一个封装了模型评估、报告打印和图表绘制功能的类。
    """

    def __init__(self, y_true, y_pred, labels, target_names):
        """
        初始化评估器。
        Args:
            y_true (list or array): 真实的标签。
            y_pred (list or array): 模型预测的标签。
            labels (list): 标签的数值列表，例如 [-1, 0, 1]。
            target_names (list): 标签对应的名称，例如 ['Negative', 'Neutral', 'Positive']。
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels
        self.target_names = target_names
        # 在初始化时就生成报告，方便后续调用
        self.report_dict = classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0
        )
        self.report_text = classification_report(
            y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
        )

    def print_report(self):
        """打印文本格式的分类报告。"""
        print("\nClassification Report:")
        print(self.report_text)

    def plot_confusion_matrix(self, save_path=None, title='混淆矩阵 (Confusion Matrix)'):
        """
        绘制并（可选地）保存混淆矩阵图。
        Args:
            save_path (Path or str, optional): 保存图表的路径. Defaults to None.
            title (str, optional): 图表标题.
        """
        cm = confusion_matrix(self.y_true, self.y_pred, labels=self.labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.target_names, yticklabels=self.target_names)
        plt.title(title)
        plt.xlabel('预测标签 (Predicted Label)')
        plt.ylabel('真实标签 (True Label)')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        plt.close()

    def plot_classification_report(self, save_path=None):
        """
        【优化】将分类报告绘制成图并（可选地）保存。
        - 移除了不适合在热力图中展示的'accuracy'行。
        - 将全局准确率添加到图表标题中。
        """
        # 从字典创建DataFrame
        report_df = pd.DataFrame(self.report_dict).T

        # CHANGED: 移除 'accuracy' 行，因为它不包含P/R/F1值
        report_df.drop('accuracy', inplace=True, errors='ignore')

        # 获取全局准确率，用于标题
        accuracy = self.report_dict.get('accuracy', 0)

        plt.figure(figsize=(10, len(report_df) * 0.6 + 2))
        sns.heatmap(report_df.iloc[:-2, :-1], annot=True, cmap='viridis', fmt='.2f', vmin=0, vmax=1)  # 不显示support列

        # CHANGED: 在标题中加入准确率信息
        plt.title(f'分类报告图 (Overall Accuracy: {accuracy:.2f})')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Classification report image saved to: {save_path}")
        plt.close()

    def plot_metrics_bar_chart(self, save_path=None):
        """
        【新增】为每个类别的 Precision, Recall, F1-score 绘制条形图。
        这个图表更直观地展示了模型在不同类别上的性能差异。
        """
        class_metrics = {name: self.report_dict[name] for name in self.target_names}

        df_metrics = pd.DataFrame({
            'Precision': [metrics['precision'] for metrics in class_metrics.values()],
            'Recall': [metrics['recall'] for metrics in class_metrics.values()],
            'F1-Score': [metrics['f1-score'] for metrics in class_metrics.values()]
        }, index=self.target_names)

        df_metrics.plot(kind='bar', figsize=(12, 7), colormap='viridis', rot=0)

        plt.title('各类别性能指标对比 (Precision, Recall, F1-Score)')
        plt.ylabel('分数 (Score)')
        plt.ylim(0, 1.1)  # Y轴范围设为0到1.1，方便查看

        # 在每个柱子上方显示数值
        for p in plt.gca().patches:
            plt.gca().annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center', xytext=(0, 9), textcoords='offset points')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Metrics bar chart saved to: {save_path}")
        plt.close()

    def get_macro_f1_score(self):
        """返回宏平均F1分数，用于最终对比。"""
        return self.report_dict['macro avg']['f1-score']