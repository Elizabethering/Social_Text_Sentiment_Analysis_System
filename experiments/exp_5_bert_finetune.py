import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
import os
from pathlib import Path
import sys

# --- 【修改】导入我们的框架 ---
sys.path.append(str(Path(__file__).parent.parent))
from utils.model_evaluator import ModelEvaluator
from config import CLASS_LABELS, CLASS_NAMES, FIGURES_DIR, PROJECT_ROOT, BERT_LABEL_MAPPING

# --- 1. 配置参数 ---
MODEL_NAME = PROJECT_ROOT / 'local_models' / 'bert-base-chinese'
OUTPUT_DIR = PROJECT_ROOT / 'models' / 'bert-finetuned-sentiment'
NUM_EPOCHS = 3
BATCH_SIZE = 16

# --- 2. 加载和预处理数据 ---
print("Loading and preparing data...")
DATA_FILE_PATH = PROJECT_ROOT / 'data' / 'combined_final_data_zn.csv'
try:
    df = pd.read_csv(DATA_FILE_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at '{DATA_FILE_PATH}'.")
    exit()

# --- START: CORRECTED CODE ---

# # Step 1: Clean the data
# df.dropna(subset=['text'], inplace=True)
# df['text'] = df['text'].astype(str)
#
# # Step 2: Map Chinese text labels to standard numeric labels (-1, 0, 1)
# # This handles the original data format.
# text_to_numeric_mapping = {'负面': -1, '中性': 0, '正面': 1}
# df['label'] = df['label'].map(text_to_numeric_mapping)
#
# # Step 3: Map the standard numeric labels to BERT-compatible labels (0, 1, 2)
# # 使用从 config.py 导入的全局配置，确保一致性
# df['labels'] = df['label'].map(BERT_LABEL_MAPPING)
#
# # Step 4: Drop any rows where mapping might have failed for any reason
# df.dropna(subset=['labels'], inplace=True)
#
# # Step 5: Ensure the final labels column is the correct integer type
# df['labels'] = df['labels'].astype(int)
#
# # --- END: CORRECTED CODE ---
# --- START: CORRECTED CODE ---

# 第1步: 清洗数据，删除'text'列为空的行
df.dropna(subset=['text'], inplace=True)
df['text'] = df['text'].astype(str)

# 第2步: 检查并清洗 'label' 列
# 确保 'label' 列中只包含 -1, 0, 1 这三个值
valid_labels = [-1, 0, 1]
df = df[df['label'].isin(valid_labels)]
df.dropna(subset=['label'], inplace=True) # 再次确认没有NaN值


df = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=min(334, len(x)), random_state=42)  # 按比例分配1000条（约334/类）
).sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱顺序并重置索引
print(f"Sampled {len(df)} records for training and validation.")


# 第3步: 将标准的数字标签 (-1, 0, 1) 映射为 BERT 兼容的标签 (0, 1, 2)
# 使用从 config.py 导入的全局配置，确保一致性
df['labels'] = df['label'].map(BERT_LABEL_MAPPING)

# 第4步: 确保'labels'列是正确的整数类型 (这一步很重要)
df['labels'] = df['labels'].astype(int)

# --- END: CORRECTED CODE ---

# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['labels'])
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
print(f"Data prepared. Training set: {len(train_dataset)}, Validation set: {len(val_dataset)}")


# --- 3. 初始化Tokenizer和模型 ---
print("Initializing tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=3,
    id2label={v: k for k, v in BERT_LABEL_MAPPING.items()}, # 使用config中的映射
    label2id={k: v for k, v in BERT_LABEL_MAPPING.items()}
)

# --- 4. 数据集Tokenization ---
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

print("Tokenizing datasets...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# --- 5. 配置训练参数并开始训练 ---
print("Setting up training...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_dir='./logs',
    logging_steps=10,
    # 核心修复：评估和保存策略必须一致
    # save_strategy="epoch",
    # evaluation_strategy="epoch",  # <-- 取消这一行的注释
    # load_best_model_at_end=True,
    # metric_for_best_model="f1",   # <-- 指定用f1分数来评判“最好”
    save_safetensors=False,
)

# 定义评估指标
import evaluate
f1_metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return f1_metric.compute(predictions=predictions, references=labels, average="macro")

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# --- 6. 保存最终模型 ---
print("Training complete. Saving the best model...")
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"Model and tokenizer saved to {OUTPUT_DIR}")


# --- 7. 【新增】使用 ModelEvaluator 进行标准化评估和绘图 ---
print("\n" + "="*50)
print("Running Standard Evaluation using ModelEvaluator")
print("="*50)

# 获取验证集的真实标签和预测结果
print("Getting predictions for standard evaluation...")
predictions = trainer.predict(tokenized_val_dataset)
y_pred_indices = np.argmax(predictions.predictions, axis=-1)
y_true_indices = predictions.label_ids


# 将BERT的标签 (0, 1, 2) 转换回项目的标准标签 (-1, 0, 1)
# 使用从 config.py 导入的全局配置来创建反向映射
reverse_label_mapping = {v: k for k, v in BERT_LABEL_MAPPING.items()}
y_pred = [reverse_label_mapping[i] for i in y_pred_indices]
y_true = [reverse_label_mapping[i] for i in y_true_indices]

# 使用 ModelEvaluator
evaluator = ModelEvaluator(
    y_true=y_true,
    y_pred=y_pred,
    labels=CLASS_LABELS,
    target_names=CLASS_NAMES
)
evaluator.print_report()

# 定义图表保存路径
cm_path = FIGURES_DIR / 'exp5_bert_confusion_matrix.png'
report_path = FIGURES_DIR / 'exp5_bert_classification_report.png'
bar_chart_path = FIGURES_DIR / 'exp5_bert_metrics_bar_chart.png'

# 调用所有绘图方法
print("\nGenerating and saving visualizations...")
evaluator.plot_confusion_matrix(save_path=cm_path, title='BERT微调模型-混淆矩阵')
evaluator.plot_classification_report(save_path=report_path)
evaluator.plot_metrics_bar_chart(save_path=bar_chart_path)

# 保存性能指标
import json
performance_file = PROJECT_ROOT / 'all_models_performance.json'
if performance_file.exists():
    with open(performance_file, 'r', encoding='utf-8') as f:
        all_performance = json.load(f)
else:
    all_performance = {}

model_name = "BERT_Finetuned"
all_performance[model_name] = evaluator.get_macro_f1_score()

with open(performance_file, 'w', encoding='utf-8') as f:
    json.dump(all_performance, f, ensure_ascii=False, indent=4)

print(f"\nPerformance for {model_name} saved.")
print("Experiment 5 finished.")