# /api/main.py (已添加CORS跨域支持)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal

from fastapi.middleware.cors import CORSMiddleware

from api.core.sentiment_predictor import get_sentiment_from_model
from api.core.keyword_extractor import get_keywords_from_engine

app = FastAPI(
    title="社交媒体舆情分析系统API",
    description="一个统一的、支持多语言情感分析和深度关键词提取的后端服务。由Elizabeth负责架构与开发。",
    version="1.0.0", # 更新版本号
)

# --- 【新增】配置CORS ---
# 定义允许访问的源列表。为了开发方便，我们使用"*"允许所有源。
# 在生产环境中，应该替换为您的前端页面的具体地址。
origins = [
    "http://localhost",
    "http://localhost:63342", # 您PyCharm启动的Web服务器地址
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "null" # 允许直接用浏览器打开的本地文件 (origin: null)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许访问的源
    allow_credentials=True, # 是否支持cookie
    allow_methods=["*"],    # 允许所有请求方法 (GET, POST, etc.)
    allow_headers=["*"],    # 允许所有请求头
)
# ----------------------


# --- 定义API的请求与响应数据格式 (不变) ---
class AnalyzeRequest(BaseModel):
    text: str
    language: Literal['chinese', 'english'] = 'chinese'

class SentimentResponse(BaseModel):
    sentiment: str
    score: float

class KeywordResponse(BaseModel):
    keyword: str
    weight: float

class AdvancedAnalysisResponse(BaseModel):
    input_text: str
    base_sentiment: SentimentResponse
    advanced_keywords: List[KeywordResponse]


# --- 创建你的API接口 (Endpoints) (不变) ---

@app.post("/analyze/sentiment", response_model=SentimentResponse, tags=["核心分析模块"])
def analyze_sentiment(request: AnalyzeRequest):
    """接收文本和语言，返回情感分析结果。"""
    result = get_sentiment_from_model(request.text, request.language)
    if not result:
        raise HTTPException(status_code=400, detail="Unsupported language")
    return result

@app.post("/analyze/advanced", response_model=AdvancedAnalysisResponse, tags=["核心分析模块"])
def analyze_advanced(request: AnalyzeRequest):
    """执行一个多步骤的高级分析流程。"""
    sentiment_result = get_sentiment_from_model(request.text, request.language)
    if not sentiment_result:
        raise HTTPException(status_code=400, detail="Unsupported language")

    keywords_result = get_keywords_from_engine(request.text, top_k=5)

    return {
        "input_text": request.text,
        "base_sentiment": sentiment_result,
        "advanced_keywords": keywords_result,
    }
