# PycharmProjects/sentiment_analysis_project/locustfile.py

import random
import pandas as pd
import os
from locust import HttpUser, task, between

# --- 1. 在测试开始前，从CSV文件加载真实的话题词 ---
# 这个部分的代码只会在Locust启动时执行一次，不会影响测试性能

print("--- Loading topics from CSV files for Locust test ---")

# 定义数据文件路径
ZH_DATA_PATH = os.path.join('data', 'combined_final_data_zn.csv')
EN_DATA_PATH = os.path.join('data', 'data_en.csv')

# 默认的话题词列表，以防文件找不到
DEFAULT_ZH_TOPICS = ['迪士尼', '酒店', '疫情', '手机', '旅游']
DEFAULT_EN_TOPICS = ['disney', 'hotel', 'movie', 'apple', 'travel']

try:
    # 使用 usecols 只读取 'topic' 列，优化内存
    zh_df = pd.read_csv(ZH_DATA_PATH, usecols=['topic_name'])
    # 获取唯一的topic并转换为列表
    ZH_TOPICS = zh_df['topic_name'].unique().tolist()
    print(f"Successfully loaded {len(ZH_TOPICS)} unique topics from Chinese dataset.")
except FileNotFoundError:
    print(f"Warning: Chinese data file not found at {ZH_DATA_PATH}. Using default topics.")
    ZH_TOPICS = DEFAULT_ZH_TOPICS

try:
    en_df = pd.read_csv(EN_DATA_PATH, usecols=['topic_name'])
    EN_TOPICS = en_df['topic_name'].unique().tolist()
    print(f"Successfully loaded {len(EN_TOPICS)} unique topics from English dataset.")
except FileNotFoundError:
    print(f"Warning: English data file not found at {EN_DATA_PATH}. Using default topics.")
    EN_TOPICS = DEFAULT_EN_TOPICS

print("--- Topic loading complete ---")
# --- 数据加载结束 ---


class AnalysisApiUser(HttpUser):
    """
    模拟一个API用户的行为，用于性能测试。
    """
    wait_time = between(1, 3)
    host = "http://127.0.0.1:8001"

    def _generate_payload(self):
        """
        随机生成一个有效的API请求体 (payload)。
        """
        language = random.choice(['zh', 'en'])

        # --- 修改部分：从加载到内存的真实话题词列表中随机选择 ---
        if language == 'zh':
            # 从中文话题词列表中选择
            topic = random.choice(ZH_TOPICS)
            model_choice = random.choice(['bert', 'snownlp'])
        else: # language == 'en'
            # 从英文话题词列表中选择
            topic = random.choice(EN_TOPICS)
            model_choice = 'bert'
        # --- 修改结束 ---

        keyword_method = random.choice([
            'textrank', 'keybert', 'hybrid',
            'pos_guided_keybert', 'hybrid_ensemble'
        ])

        payload = {
            "language": language,
            "topic": topic,
            "start_date": "2024-07-01",
            "end_date": "2024-08-01",
            "model_choice": model_choice,
            "keyword_method": keyword_method,
            "top_k_keywords": 50
        }
        return payload

    @task
    def analyze_endpoint(self):
        """
        定义一个任务，用于测试 /analyze 端点。
        """
        random_payload = self._generate_payload()
        headers = {"Content-Type": "application/json"}

        with self.client.post(
            "/analyze",
            json=random_payload,
            headers=headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code was {response.status_code}, expected 200. Response text: {response.text}")