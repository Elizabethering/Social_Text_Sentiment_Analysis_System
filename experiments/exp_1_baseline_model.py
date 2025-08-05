# /experiments/exp_1_baseline_model.py (诊断版)

import pandas as pd
from snownlp import SnowNLP

# 从我们自己的模块中导入工具函数
from utils.data_loader import load_weibo_senti_data
from utils.model_evaluator import evaluate_and_report, plot_and_save_confusion_matrix


def run_baseline_experiment():
    """执行并评估SnowNLP基线模型"""
    print("--- 开始执行实验一：SnowNLP基线模型 (诊断模式) ---")

    # 1. 加载数据
    df = load_weibo_senti_data()

    # --- 【诊断点 1】: 检查数据是否加载成功 ---
    if df.empty:
        print("【诊断结果】: 数据加载失败，DataFrame为空。请检查data_loader.py或数据文件路径。脚本终止。")
        return
    print(f"【诊断信息】: 数据加载成功，共 {len(df)} 条记录。")
    print("【诊断信息】: 数据集前3行内容:")
    print(df.head(3))
    # -----------------------------------------

    # 为了快速演示，我们可以只取一部分样本
    df_sample = df.copy()
    print(f"\n使用 {len(df_sample)} 条样本进行本次实验...")

    # 2. 定义预测函数
    def predict_sentiment_by_snownlp(text):
        try:
            if not isinstance(text, str) or not text.strip():
                return 0.5
            return SnowNLP(text).sentiments
        except Exception as e:
            print(f"【诊断警告】: SnowNLP处理文本时发生异常: {e}")
            return 0.5

    # 3. 进行预测
    print("\n正在使用SnowNLP进行情感预测...")
    # --- 【诊断点 2】: 检查列名是否正确 ---
    if 'review' not in df_sample.columns:
        print("【诊断结果】: 错误！数据集中未找到 'review' 列。请检查CSV文件列名。脚本终止。")
        return
    # ------------------------------------
    df_sample['predicted_score'] = df_sample['review'].apply(predict_sentiment_by_snownlp)
    df_sample['predicted_label'] = (df_sample['predicted_score'] >= 0.5).astype(int)
    print("预测完成。")
    print("【诊断信息】: 预测结果预览:")
    print(df_sample[['review', 'label', 'predicted_label']].head(3))

    # 4. 评估模型性能
    # --- 【诊断点 3】: 检查标签列是否存在 ---
    if 'label' not in df_sample.columns:
        print("【诊断结果】: 错误！数据集中未找到 'label' 列。请检查CSV文件列名。脚本终止。")
        return
    # -------------------------------------
    y_true = df_sample['label']
    y_pred = df_sample['predicted_label']

    labels = [0, 1]
    target_names = ['负面 (Negative)', '正面 (Positive)']

    print("\n准备调用模型评估工具...")
    evaluate_and_report(y_true, y_pred, labels, target_names)

    # --- 【诊断点 4】: 检查保存路径和最终数据 ---
    save_path = "reports/figures/baseline_model_confusion_matrix.png"
    print(f"【诊断信息】: 即将调用绘图函数，准备将图片保存至: {save_path}")
    print(f"【诊断信息】: 传入绘图函数的真实标签有 {len(y_true)} 个，预测标签有 {len(y_pred)} 个。")
    # ----------------------------------------

    # 调用绘图函数
    try:
        plot_and_save_confusion_matrix(y_true, y_pred, labels, target_names, save_path)
    except Exception as e:
        print(f"【诊断结果】: 调用绘图函数时发生未知错误: {e}")

    print("\n--- 实验一结束 ---")


if __name__ == '__main__':
    run_baseline_experiment()