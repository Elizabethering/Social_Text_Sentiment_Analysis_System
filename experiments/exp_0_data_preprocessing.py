# experiments/exp_0_data_preprocessing.py (Corrected Version)
import pandas as pd
from sklearn.model_selection import train_test_split
import re


def clean_text(text):
    """一个简单的文本清洗函数"""
    # 确保输入是字符串
    if not isinstance(text, str):
        return ""
    # 移除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # 移除@提及
    text = re.sub(r'@[\w_]+', '', text)
    # 移除#话题#
    text = re.sub(r'#.*?#', '', text)
    # 移除多余的空格和换行
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def prepare_data(file_path='../data/weibo_senti_100k.csv'):
    """
    读取、清洗并划分微博数据集。
    """
    print("--- 开始执行【数据预处理】 ---")

    # 1. 读取数据
    try:
        df = pd.read_csv(file_path, header=None, names=['label', 'text'])
        print(f"成功从 '{file_path}' 加载了 {len(df)} 条原始数据。")
    except FileNotFoundError:
        print(f"错误：找不到数据文件 '{file_path}'。请确保你已经下载了数据并放在了 'data/' 目录下。")
        return

    # 2. 清洗文本
    print("正在清洗文本...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # 3. 移除空数据
    df.dropna(subset=['cleaned_text'], inplace=True)
    df = df[df['cleaned_text'] != '']
    print(f"清洗后，剩余 {len(df)} 条有效数据。")


    # 4. 在划分前，检查并处理样本数过少的类别
    print("\n正在检查每个类别的样本数量...")
    label_counts = df['label'].value_counts()

    # 找出样本数少于2的类别
    rare_labels = label_counts[label_counts < 2].index

    if not rare_labels.empty:
        print(f"警告：发现样本数极少的类别: {list(rare_labels)}。这些类别的样本将被移除以保证数据划分的顺利进行。")
        # 从df中移除这些孤独类别的样本
        original_count = len(df)
        df = df[~df['label'].isin(rare_labels)]
        print(f"移除了 {original_count - len(df)} 条样本。当前剩余数据量: {len(df)}")
    else:
        print("数据集中所有类别的样本数均大于等于2，无需处理。")

    # 5. 划分训练集和测试集
    print("\n正在划分训练集和测试集...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,  # 设置随机种子以保证结果可复现
        stratify=df['label']  # stratify确保划分后正负样本比例不变
    )

    # 6. 保存划分好的数据
    train_df.to_csv('../data/weibo_train_data.csv', index=False, encoding='utf-8-sig')
    test_df.to_csv('../data/weibo_test_data.csv', index=False, encoding='utf-8-sig')

    print("\n--- 【数据预处理】执行完毕！ ---")
    print(f"总数据量: {len(df)}")
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print("文件 'weibo_train_data.csv' 和 'weibo_test_data.csv' 已保存至 'data/' 目录下。")


if __name__ == '__main__':
    prepare_data()