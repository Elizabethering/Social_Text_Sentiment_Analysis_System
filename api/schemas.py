# # sentiment_analysis_project/api/schemas.py
#
# from typing import List, Dict, Any, Literal
# from pydantic import BaseModel, Field
#
# # --- Request Model ---
# class AnalysisRequest(BaseModel):
#     language: Literal['zh', 'en'] = Field(..., description="分析语言选择: 'zh' (中文) 或 'en' (英文)")
#     topic: str = Field(..., description="要分析的话题关键词")
#     start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)", pattern=r"^\d{4}-\d{2}-\d{2}$")
#     end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)", pattern=r"^\d{4}-\d{2}-\d{2}$")
#     model_choice: Literal['bert', 'snownlp'] = Field(..., description="情感分析模型")
#     keyword_method: Literal['textrank', 'keybert', 'hybrid', 'pos_guided_keybert', 'hybrid_ensemble'] = Field(
#         ...,
#         description="关键词提取算法"
#     )
#     top_k_keywords: int = Field(..., description="返回的关键词数量", ge=1)
#
#     class Config:
#         json_schema_extra = {
#             "example": {
#                 "language": "zh",
#                 "topic": "迪士尼",
#                 "start_date": "2024-07-01",
#                 "end_date": "2024-08-01",
#                 "model_choice": "bert",
#                 "keyword_method": "hybrid_ensemble",
#                 "top_k_keywords": 50
#             }
#         }
#
# # --- Response Models ---
# class SentimentDistribution(BaseModel):
#     positive: int
#     neutral: int
#     negative: int
#
# class SentimentTimePoint(BaseModel):
#     date: str
#     positive_count: int
#     negative_count: int
#     neutral_count: int
#
# class TopicHotnessPoint(BaseModel):
#     date: str
#     topic: str
#     hotness: int
#
# class KeywordItem(BaseModel):
#     word: str
#     weight: float
#
# class AnalysisResponse(BaseModel):
#     topic_info: Dict[str, Any]
#     sentiment_pie_chart: SentimentDistribution
#     sentiment_trend_chart: List[SentimentTimePoint]
#     topic_word_cloud: List[KeywordItem]
#     hot_topics_trend_chart: List[TopicHotnessPoint]
# sentiment_analysis_project/api/schemas.py

# sentiment_analysis_project/api/schemas.py

from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field


# --- Request Model (请求模型) ---
class AnalysisRequest(BaseModel):
    language: Literal['zh', 'en'] = Field(..., description="分析语言选择: 'zh' (中文) 或 'en' (英文)")
    topic: str = Field(..., description="要分析的话题关键词")
    start_date: str = Field(..., description="开始日期 (YYYY-MM-DD)", pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str = Field(..., description="结束日期 (YYYY-MM-DD)", pattern=r"^\d{4}-\d{2}-\d{2}$")
    model_choice: Literal['bert', 'snownlp'] = Field(..., description="情感分析模型")
    keyword_method: Literal['textrank', 'keybert', 'hybrid', 'pos_guided_keybert', 'hybrid_ensemble'] = Field(
        ...,
        description="关键词提取算法"
    )
    top_k_keywords: int = Field(..., description="返回的关键词数量", ge=1)


# --- Response Models (响应模型) ---
class SentimentDistribution(BaseModel):
    positive: int
    neutral: int
    negative: int


class SentimentTimePoint(BaseModel):
    date: str
    positive_count: int
    negative_count: int
    neutral_count: int


class TopicHotnessPoint(BaseModel):
    date: str
    topic: str
    hotness: int


class KeywordItem(BaseModel):
    word: str
    weight: float


class AnalysisResponse(BaseModel):
    topic_info: Dict[str, Any]
    sentiment_pie_chart: SentimentDistribution
    sentiment_trend_chart: List[SentimentTimePoint]

    # 【核心修改】将原有的词云重命名，并增加正面和负面关键词列表
    overall_word_cloud: List[KeywordItem] = Field(..., description="数据源: 话题综合关键词词云图。")
    positive_keywords: List[KeywordItem] = Field(..., description="数据源: 正面评价中的核心关键词。")
    negative_keywords: List[KeywordItem] = Field(..., description="数据源: 负面评价中的核心关键词。")

    hot_topics_trend_chart: List[TopicHotnessPoint]