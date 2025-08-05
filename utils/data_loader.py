# /utils/data_loader.py (修改版)

import pandas as pd
from pathlib import Path

# 获取项目根目录的路径
# Path(__file__) 是当前文件 (data_loader.py) 的路径
# .resolve() 将其转为绝对路径
# .parent.parent 会向上跳两级 (从 utils -> 项目根目录)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_weibo_senti_data(file_name: str = "weibo_senti_100k.csv") -> pd.DataFrame:
    """
    加载微博情感数据集。
    现在它会从项目根目录开始寻找数据文件。
    """
    # 将项目根目录和 'data' 文件夹以及文件名拼接起来
    data_path = PROJECT_ROOT / "data" / file_name

    if not data_path.exists():
        print(f"错误: 数据文件未找到 at {data_path}")
        return pd.DataFrame()

    print(f"正在从 {data_path} 加载数据...")
    df = pd.read_csv(data_path)
    print(f"数据加载完成，共 {len(df)} 条记录。")
    return df

# ... (测试代码部分不变) ...