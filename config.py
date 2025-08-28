from pathlib import Path

# --- 根目录 ---
# 项目的根目录
PROJECT_ROOT = Path(__file__).parent

# --- 数据路径 ---
DATA_DIR = PROJECT_ROOT / 'data'
# 数据文件路径
DATA_FILE = DATA_DIR / 'combined_final_data_zn.csv'

# --- 输出路径 ---
# 存放所有报告和图表的文件夹
REPORTS_DIR = PROJECT_ROOT / 'reports'
# 存放所有实验生成的图表的文件夹
FIGURES_DIR = REPORTS_DIR / 'figures'
# 确保图表文件夹存在
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- 模型评估参数 ---
# 定义情感标签和对应的名称，在所有实验中保持一致
# 我们的标准：-1:负面, 0:中性, 1:正面
CLASS_LABELS = [-1, 0, 1]
CLASS_NAMES = ['Negative', 'Neutral', 'Positive']
# BERT模型在训练时使用的标签映射: -1->0, 0->1, 1->2
BERT_LABEL_MAPPING = {-1: 0, 0: 1, 1: 2}