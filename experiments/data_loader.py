import pandas as pd
import sys
import os
import random
from datetime import datetime, timedelta, timezone
from sklearn.model_selection import train_test_split

# --- 配置 ---
# 确保此脚本可以找到项目根目录，以便正确引用其他文件
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
sys.path.append(PROJECT_ROOT)

# --- 数据文件路径定义 ---
HOTEL_DATA_FILE = os.path.join(DATA_DIR, 'ChnSentiCorp_htl_all.csv')
SHOPPING_DATA_FILE = os.path.join(DATA_DIR, 'online_shopping_10_cats.csv')
DATA_ZN_FILE = os.path.join(DATA_DIR, 'data_zn.csv')
# 【新增】电影评论数据文件路径，请根据您的实际文件名修改
MOVIE_DATA_FILE = os.path.join(DATA_DIR, 'douban_movie_short_comments.csv')  # 假设文件名为这个，如果不是请修改

# --- 输出文件 ---
FINAL_COMBINED_FILE = os.path.join(DATA_DIR, 'combined_final_data_zn.csv')


# --- 辅助函数 ---

def generate_random_timestamp(start_date, end_date):
    """
    在一个给定的日期范围内生成一个带时区的随机时间戳。
    格式: YYYY-MM-DD HH:MM:SS+08:00
    """
    time_diff = end_date - start_date
    random_seconds = random.randint(0, int(time_diff.total_seconds()))
    random_date = start_date + timedelta(seconds=random_seconds)
    # 设置时区为 UTC+8
    tz = timezone(timedelta(hours=8))
    return random_date.astimezone(tz).strftime('%Y-%m-%d %H:%M:%S%z')


def standardize_label(label):
    """
    将不同的标签格式统一为数字：1 (正面), 0 (中性), -1 (负面)。
    """
    if pd.isna(label):
        return None

    label_str = str(label).strip().lower()

    if label_str in ['正面', '1', '1.0']:
        return 1
    elif label_str in ['负面', '0', '0.0']:
        return -1
    elif label_str in ['中性']:
        return 0
    return None


# --- 核心处理函数 ---

def process_hotel_data(file_path):
    """加载并处理酒店评论数据。"""
    print(f"  - Loading {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    df.rename(columns={'review': 'text'}, inplace=True)
    df['topic_name'] = '酒店'
    start_date = datetime(2020, 8, 1)
    end_date = datetime(2025, 8, 15, 23, 59, 59)
    df['time'] = [generate_random_timestamp(start_date, end_date) for _ in range(len(df))]
    return df[['label', 'topic_name', 'text', 'time']]


def process_shopping_data(file_path):
    """加载并处理电商评论数据。"""
    print(f"  - Loading {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    df.rename(columns={'review': 'text', 'cat': 'topic_name'}, inplace=True)
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2025, 8, 31, 23, 59, 59)
    df['time'] = [generate_random_timestamp(start_date, end_date) for _ in range(len(df))]
    return df[['label', 'topic_name', 'text', 'time']]


def process_hierarchical_data(file_path):
    """加载并处理具有层级结构的 data_zn.csv 数据。"""
    print(f"  - Loading and processing hierarchical data from {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    text_map = df[df['type'] == 0].set_index('text_id')['text'].to_dict()
    processed_rows = []

    for _, row in df.iterrows():
        if row['type'] == 0:
            processed_rows.append({
                'label': row['label'], 'topic_name': row['topic_name'],
                'text': row['text'], 'time': row['time']
            })
        else:
            parent_text = text_map.get(row['prior_id'], '')
            if parent_text:
                combined_text = f"{parent_text} [回复] {row['text']}"
                processed_rows.append({
                    'label': row['label'], 'topic_name': row['topic_name'],
                    'text': combined_text, 'time': row['time']
                })
    return pd.DataFrame(processed_rows)


def process_movie_data(file_path):
    """【已更新】加载并处理电影评论数据，保留原始日期，随机化时间。"""
    print(f"  - Loading {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)

    def convert_star_to_label(star):
        if star in [1, 2]:
            return -1
        elif star == 3:
            return 0
        elif star in [4, 5]:
            return 1
        return None

    def convert_date_to_timestamp(date_str):
        if pd.isna(date_str): return None
        try:
            base_date = datetime.strptime(str(date_str), '%Y-%m-%d')
            random_time = timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59),
                                    seconds=random.randint(0, 59))
            full_datetime = base_date + random_time
            tz = timezone(timedelta(hours=8))
            return full_datetime.astimezone(tz).strftime('%Y-%m-%d %H:%M:%S%z')
        except (ValueError, TypeError):
            return None

    df['label'] = df['Star'].apply(convert_star_to_label)
    df['time'] = df['Date'].apply(convert_date_to_timestamp)
    df.rename(columns={'Comment': 'text', 'Movie_Name_CN': 'topic_name'}, inplace=True)
    return df[['label', 'topic_name', 'text', 'time']]


# --- 主函数 ---

def combine_new_datasets():
    """加载、处理并合并所有四个数据集。"""
    print("\n" + "=" * 50)
    print("Step 1: Starting Data Combination Process")
    print("=" * 50)

    all_processed_dfs = []

    # 依次处理所有数据集
    try:
        df_hotel = process_hotel_data(HOTEL_DATA_FILE)
        all_processed_dfs.append(df_hotel)
        print(f"  - Processed {len(df_hotel)} rows from hotel data.")
    except FileNotFoundError:
        print(f"  - WARNING: Hotel data not found at {HOTEL_DATA_FILE}. Skipping.")

    try:
        df_shopping = process_shopping_data(SHOPPING_DATA_FILE)
        all_processed_dfs.append(df_shopping)
        print(f"  - Processed {len(df_shopping)} rows from shopping data.")
    except FileNotFoundError:
        print(f"  - WARNING: Shopping data not found at {SHOPPING_DATA_FILE}. Skipping.")

    try:
        df_zn = process_hierarchical_data(DATA_ZN_FILE)
        all_processed_dfs.append(df_zn)
        print(f"  - Processed {len(df_zn)} rows from hierarchical data.")
    except FileNotFoundError:
        print(f"  - WARNING: Hierarchical data not found at {DATA_ZN_FILE}. Skipping.")

    try:
        df_movie = process_movie_data(MOVIE_DATA_FILE)
        all_processed_dfs.append(df_movie)
        print(f"  - Processed {len(df_movie)} rows from movie data.")
    except FileNotFoundError:
        print(f"  - WARNING: Movie data not found at {MOVIE_DATA_FILE}. Skipping.")

    if not all_processed_dfs:
        print("\nFATAL: No data files were loaded. Aborting.")
        return

    # 合并与最终清洗
    print("\n" + "-" * 50)
    print("Combining all processed dataframes...")
    combined_df = pd.concat(all_processed_dfs, ignore_index=True)
    print(f"Total rows before final cleaning: {len(combined_df)}")

    combined_df['label'] = combined_df['label'].apply(standardize_label)
    combined_df.dropna(subset=['text', 'label', 'topic_name', 'time'], inplace=True)
    combined_df = combined_df[combined_df['text'].str.strip() != '']
    combined_df.drop_duplicates(subset=['text'], keep='first', inplace=True)
    combined_df['label'] = combined_df['label'].astype(int)
    combined_df.reset_index(drop=True, inplace=True)

    print(f"Total rows in final dataset: {len(combined_df)}")

    # 保存文件
    try:
        print(f"\nSaving final combined data to: {FINAL_COMBINED_FILE}")
        combined_df.to_csv(FINAL_COMBINED_FILE, index=False, encoding='utf-8-sig')
        print("  - Save successful!")
    except Exception as e:
        print(f"  - ERROR: Failed to save file. Reason: {e}")
    print("=" * 50)
    print("Step 1 Finished Successfully")
    print("=" * 50)


# --- 【新增】加载并划分数据集的函数 ---
def load_and_split_data(data_file=FINAL_COMBINED_FILE, test_size=0.2, random_state=42):
    """
    加载已合并清洗的数据集，并将其划分为训练集和测试集。
    """
    print("\n" + "=" * 50)
    print("Step 2: Loading Combined Data and Splitting")
    print("=" * 50)

    try:
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        print(f"Successfully loaded {len(df)} rows.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file not found at {data_file}")
        print("Please run the data combination step first.")
        return None, None, None, None

    # 划分特征和标签
    X = df['text']
    y = df['label']

    print(f"Splitting data into training ({1 - test_size:.0%}) and testing ({test_size:.0%}) sets...")

    # 使用分层抽样 (stratify=y) 确保标签分布在训练集和测试集中保持一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Splitting complete.")
    print(f"  - Training set size: {len(X_train)}")
    print(f"  - Testing set size:  {len(X_test)}")
    print("=" * 50)
    print("Step 2 Finished Successfully")
    print("=" * 50)

    return X_train, X_test, y_train, y_test


# --- 主执行入口 ---
if __name__ == '__main__':
    # 流程第一步：合并所有原始数据集，生成一个大的、干净的CSV文件
    combine_new_datasets()

    # 流程第二步：加载上一步生成的CSV文件，并将其划分为训练集和测试集
    # 在实际项目中，你可能会在另一个脚本（如 train.py）中调用这个函数
    X_train, X_test, y_train, y_test = load_and_split_data()

    # 检查返回结果是否有效
    if X_train is not None:
        print("\nData is ready for model training.")
        print(f"Example training text: '{X_train.iloc[0][:50]}...'")
        print(f"Example training label: {y_train.iloc[0]}")

# import pandas as pd
# from sklearn.model_selection import train_test_split
# import sys
# import os
#
# # 导入我们的配置
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from config import DATA_FILE
#
#
# def clean_data(df):
#     """
#     一个专门用于清洗和预处理原始数据的函数。
#     """
#     print("Cleaning and preprocessing raw data...")
#
#     # 步骤1：重命名列，将 'topic_name' 改为 'topic'
#     if 'topic_name' in df.columns:
#         df.rename(columns={'topic_name': 'topic'}, inplace=True)
#         print("  - Renamed column 'topic_name' to 'topic'.")
#
#     # 步骤2：处理缺失的 'text' 数据，直接删除这些行
#     initial_rows = len(df)
#     df.dropna(subset=['text'], inplace=True)
#     print(f"  - Dropped {initial_rows - len(df)} rows with missing text.")
#
#     # 步骤3：将文字标签映射为数字标签
#     if 'label' in df.columns and df['label'].dtype == 'object':
#         label_mapping = {'负面': -1, '中性': 0, '正面': 1}
#         df['label'] = df['label'].map(label_mapping)
#         print("  - Mapped text labels ('负面', '中性', '正面') to numeric labels (-1, 0, 1).")
#
#     # 步骤4：确保关键列的数据类型正确
#     df['text'] = df['text'].astype(str)
#     df['label'] = df['label'].astype(int)
#
#     print("Data cleaning complete.")
#     return df
#
#
# def load_and_split_data(test_size=0.2, random_state=42):
#     """
#     加载数据，进行清洗，然后划分为训练集和测试集。
#     """
#     print("\n" + "-" * 30)
#     print("Loading data...")
#     try:
#         # 加载原始数据
#         df_raw = pd.read_csv(DATA_FILE)
#     except FileNotFoundError:
#         print(f"Error: Data file not found at {DATA_FILE}")
#         return None, None, None, None
#
#     df_cleaned = clean_data(df_raw)
#
#     # 使用清洗后的数据进行后续操作
#     X = df_cleaned['text']
#     y = df_cleaned['label']
#
#     print("Splitting data into training and testing sets...")
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state, stratify=y
#     )
#
#     print(f"Data loading process finished. Training set: {len(X_train)} samples, Testing set: {len(X_test)} samples.")
#     print("-" * 30 + "\n")
#     return X_train, X_test, y_train, y_test
#
#
# if __name__ == '__main__':
#     # 测试新的数据加载和清洗流程
#     load_and_split_data()
