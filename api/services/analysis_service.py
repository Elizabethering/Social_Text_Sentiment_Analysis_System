# api/services/analysis_service.py

import logging
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
from collections import defaultdict

# 导入核心模块
from ..core.sentiment_predictor import SentimentPredictor
from ..core.keyword_extractor import KeywordExtractor
# 导入 Pydantic 模型
from ..schemas import (
    AnalysisRequest, AnalysisResponse, SentimentDistribution,
    SentimentTimePoint, TopicHotnessPoint, KeywordItem
)

# 获取日志记录器
logger = logging.getLogger(__name__)


class DataService:
    """
    一个专门负责加载和提供数据集的服务类。
    """

    def __init__(self, zh_path: str, en_path: str):
        """
        在服务实例化时，预加载所有数据。
        """
        logger.info(f"正在初始化 DataService...")
        self.df_zh = self._load_data(zh_path, "中文")
        self.df_en = self._load_data(en_path, "英文")

    def _load_data(self, path: str, lang_name: str) -> pd.DataFrame:
        """
        从指定的CSV路径加载数据，并将其解析为DataFrame。
        """
        try:
            # 确保 'time' 列被正确解析为带时区的日期时间对象
            df = pd.read_csv(path, parse_dates=['time'])
            logger.info(f"{lang_name}数据集加载成功，共 {len(df)} 条记录")
            return df
        except FileNotFoundError:
            logger.warning(f"警告: {lang_name}数据集未找到 (路径: {path})")
            return pd.DataFrame(columns=['topic_name', 'text', 'time', 'label'])
        except Exception as e:
            logger.error(f"加载 {lang_name} 数据集时出错: {e}")
            return pd.DataFrame(columns=['topic_name', 'text', 'time', 'label'])

    def get_df(self, language: str) -> pd.DataFrame:
        """
        根据语言选择，返回对应的DataFrame。
        """
        if language == 'zh':
            return self.df_zh
        elif language == 'en':
            return self.df_en
        return pd.DataFrame()


class AnalysisService:
    """
    执行核心分析任务的服务。
    """

    def __init__(
            self,
            data_service: DataService,
            predictor: SentimentPredictor,
            extractor: KeywordExtractor
    ):
        self.data_service = data_service
        self.sentiment_predictor = predictor
        self.keyword_extractor = extractor

    def run_analysis(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        执行完整的分析流程，包含情感归因。
        """
        df_all = self.data_service.get_df(request.language)

        if df_all.empty:
            logger.warning("请求的语言对应的数据集为空，无法进行分析。")
            # 返回一个空的响应结构，避免前端报错
            return self._create_empty_response(request.topic)

        # 1. 修正时区问题：将前端传来的“天真”日期转换为与数据一致的“感知时区”日期
        try:
            target_timezone = df_all['time'].dt.tz
            if target_timezone is None:  # 如果数据本身没有时区，则假定为UTC+8
                target_timezone = timezone('Asia/Shanghai')

            start_dt = timezone('Asia/Shanghai').localize(datetime.fromisoformat(request.start_date)).astimezone(
                target_timezone)
            end_dt = (timezone('Asia/Shanghai').localize(datetime.fromisoformat(request.end_date)) + timedelta(
                days=1)).astimezone(target_timezone)
        except Exception as e:
            logger.error(f"日期转换失败: {e}. 使用无时区比较。")
            start_dt = datetime.fromisoformat(request.start_date)
            end_dt = datetime.fromisoformat(request.end_date) + timedelta(days=1)

        # 2. 过滤数据
        period_df = df_all[(df_all['time'] >= start_dt) & (df_all['time'] < end_dt)].copy()
        topic_df = period_df[period_df['topic_name'] == request.topic].copy()

        if topic_df.empty:
            return self._create_empty_response(request.topic)

        # 3. 基础情感分析
        topic_df['sentiment'] = topic_df['text'].apply(
            lambda x: self.sentiment_predictor.predict(x, model=request.model_choice, lang=request.language)
            if pd.notna(x) else 0
        )

        # 4. 计算图表数据 (饼图和趋势图)
        pie_data, trend_data = self._calculate_sentiment_stats(topic_df)

        # 5. 综合关键词提取
        overall_keywords = self.keyword_extractor.extract(
            texts=topic_df['text'].dropna().tolist(),
            lang=request.language,
            top_k=request.top_k_keywords,
            method=request.keyword_method
        )

        # 6. 【核心功能】情感联动关键词分析 (情感归因)
        positive_keywords_final, negative_keywords_final = self._get_sentiment_linked_keywords(topic_df, request)

        # 7. 热点图计算
        hot_data = self._calculate_hot_topics(period_df)

        # 8. 构建最终响应
        return AnalysisResponse(
            topic_info={"topic": request.topic, "total_texts": len(topic_df)},
            sentiment_pie_chart=pie_data,
            sentiment_trend_chart=sorted(trend_data, key=lambda x: x.date),
            overall_word_cloud=overall_keywords,
            positive_keywords=positive_keywords_final,
            negative_keywords=negative_keywords_final,
            hot_topics_trend_chart=hot_data
        )

    def _create_empty_response(self, topic: str) -> AnalysisResponse:
        """创建一个空的响应对象，用于无数据时返回。"""
        return AnalysisResponse(
            topic_info={"topic": topic, "total_texts": 0},
            sentiment_pie_chart=SentimentDistribution(positive=0, neutral=0, negative=0),
            sentiment_trend_chart=[],
            overall_word_cloud=[],
            positive_keywords=[],
            negative_keywords=[],
            hot_topics_trend_chart=[]
        )

    def _calculate_sentiment_stats(self, topic_df: pd.DataFrame):
        """计算情感分布饼图和趋势图的数据。"""
        counts = topic_df['sentiment'].value_counts().to_dict()
        pie_data = SentimentDistribution(
            positive=counts.get(1, 0),
            neutral=counts.get(0, 0),
            negative=counts.get(-1, 0)
        )

        topic_df['date'] = topic_df['time'].dt.date
        trend_df = topic_df.groupby('date')['sentiment'].value_counts().unstack(fill_value=0)
        trend_data = [
            SentimentTimePoint(
                date=d.strftime('%Y-%m-%d'),
                positive_count=r.get(1, 0),
                negative_count=r.get(-1, 0),
                neutral_count=r.get(0, 0)
            ) for d, r in trend_df.iterrows()
        ]
        return pie_data, trend_data

    def _get_sentiment_linked_keywords(self, topic_df: pd.DataFrame, request: AnalysisRequest):
        """提取正面和负面文本中的关键词。"""
        logger.info("Starting sentiment-keyword linkage analysis...")

        positive_texts = topic_df[topic_df['sentiment'] == 1]['text'].dropna().tolist()
        negative_texts = topic_df[topic_df['sentiment'] == -1]['text'].dropna().tolist()

        positive_keywords_final = []
        if positive_texts:
            positive_keywords_final = self.keyword_extractor.extract(
                texts=positive_texts,
                lang=request.language,
                top_k=request.top_k_keywords,
                method=request.keyword_method
            )

        negative_keywords_final = []
        if negative_texts:
            negative_keywords_final = self.keyword_extractor.extract(
                texts=negative_texts,
                lang=request.language,
                top_k=request.top_k_keywords,
                method=request.keyword_method
            )

        logger.info("Linkage analysis complete.")
        return positive_keywords_final, negative_keywords_final

    def _calculate_hot_topics(self, period_df: pd.DataFrame):
        """计算全时段的热点话题趋势。"""
        period_df['date'] = period_df['time'].dt.date
        hot_df = period_df.groupby(['date', 'topic_name']).size().reset_index(name='hotness')
        hot_df.rename(columns={'topic_name': 'topic'}, inplace=True)
        hot_df['date'] = hot_df['date'].apply(lambda d: d.strftime('%Y-%m-%d'))
        return [TopicHotnessPoint(**row) for row in hot_df.to_dict('records')]

