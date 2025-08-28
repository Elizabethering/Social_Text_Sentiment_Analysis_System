# # sentiment_analysis_project/api/main.py
#
# import logging
# import os
# import sys
# from contextlib import asynccontextmanager
#
# import pandas as pd
# from adodbapi.setup import AUTHOR
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
#
# # --- 1. 初始化与路径配置 ---
# print("--- 正在执行最新的 main.py 文件 ---")  # 调试语句
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(PROJECT_ROOT)
#
# # --- 2. 导入所有需要的模块 ---
# from api.core.sentiment_predictor import SentimentPredictor
# from api.core.keyword_extractor import KeywordExtractor
# from api.services.analysis_service import AnalysisService, DataService
# from api.schemas import AnalysisRequest, AnalysisResponse
#
#
# # --- 3. 应用配置与全局状态 ---
# class AppConfig:
#     """应用配置"""
#     ZH_DATA_PATH: str = os.path.join(PROJECT_ROOT, 'data', 'combined_final_data_zn.csv')
#     EN_DATA_PATH: str = os.path.join(PROJECT_ROOT, 'data', 'data_en.csv')
#     API_TITLE: str = "舆情情感分析API"
#     API_VERSION: str = "4.0.0 "
#     AUTHOR: str = "Dora"
#
# class AppState:
#     """应用全局状态，存放加载后的实例"""
#     df_zh: pd.DataFrame
#     df_en: pd.DataFrame
#     analysis_service: AnalysisService
#
#
# app_state = AppState()
#
#
# def _load_data(path: str, lang_name: str) -> pd.DataFrame:
#     """辅助函数：加载数据文件"""
#     try:
#         df = pd.read_csv(path, parse_dates=['time'])
#         logger.info(f"{lang_name}数据集加载成功，共 {len(df)} 条记录")
#         return df
#     except FileNotFoundError:
#         logger.warning(f"警告: {lang_name}数据集未找到 (路径: {path})")
#         return pd.DataFrame()
#
#
# # --- 4. FastAPI应用与生命周期事件 ---
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Application lifecycle manager. Loads models and data on startup.
#     """
#     logger.info("Application starting up... loading models and data...")
#
#     # 1. First, create the DataService instance
#     app_state.data_service = DataService(AppConfig.ZH_DATA_PATH, AppConfig.EN_DATA_PATH)
#
#     # 2. Initialize the other models
#     sentiment_predictor = SentimentPredictor()
#     keyword_extractor = KeywordExtractor()
#
#     # 3. CORRECTED: Pass the data_service instance when creating AnalysisService
#     app_state.analysis_service = AnalysisService(
#         data_service=app_state.data_service,  # <-- This was the missing part
#         predictor=sentiment_predictor,
#         extractor=keyword_extractor
#     )
#
#     logger.info("API is ready, all models and data are pre-loaded.")
#     yield
#     logger.info("Application shutting down...")
#
# # 定义联系人信息，这是 OpenAPI 规范的一部分
# contact_info = {
#     "name": AppConfig.AUTHOR,
#     "email": ".com", # 可选：您的邮箱
# }
#
# print(f"--- FastAPI 应用即将创建，版本号: {AppConfig.API_VERSION} ---")  # 调试语句
# app = FastAPI(
#
#     title=AppConfig.API_TITLE,
#     version=AppConfig.API_VERSION,
#     lifespan=lifespan,  # 在这里注册生命周期管理器
#     contact=contact_info # 正确地将联系人信息传入这里
# )
#
# # --- 修正CORS配置 ---
# # 定义一个允许的源列表
# origins = [
#     "http://localhost:63342",  # 明确允许来自PyCharm内置服务器的请求
#     "http://127.0.0.1:63342",
#     # "http://localhost:8001",
#     # "http://127.0.0.1:8001",
#     "null" # 允许直接通过 file:/// 打开html文件
# ]
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,      # <--- 使用我们上面定义的origins列表
#     allow_credentials=True,
#     allow_methods=["*"],        # 允许所有HTTP方法
#     allow_headers=["*"],        # 允许所有HTTP头
# )
#
#
# # --- 5. API 端点 ---
# @app.post("/analyze", response_model=AnalysisResponse, summary="执行舆情分析")
# async def analyze_topic(request: AnalysisRequest):
#     """接收请求，并将其委托给 AnalysisService 进行处理。"""
#     # 只需要传递 request，服务自己会处理数据获取
#     return app_state.analysis_service.run_analysis(request=request)
#
# @app.get("/", tags=["系统"])
# async def root():
#     """提供一个简单的健康检查端点。"""
#     return {"message": "舆情分析API已启动", "docs_url": "/docs"}
#
#
# # --- 6. 应用启动 (用于直接运行调试) ---
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
# sentiment_analysis_project/api/main.py

import logging
import os
import sys
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- 1. 初始化与路径配置 ---
print("--- 正在执行最新的 main.py 文件 ---")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# --- 2. 导入所有需要的模块 ---
# 确保您的项目结构与此匹配
from api.core.sentiment_predictor import SentimentPredictor
from api.core.keyword_extractor import KeywordExtractor
from api.services.analysis_service import AnalysisService, DataService
from api.schemas import AnalysisRequest, AnalysisResponse


# --- 3. 应用配置与全局状态 ---
class AppConfig:
    """应用配置"""
    ZH_DATA_PATH: str = os.path.join(PROJECT_ROOT, 'data', 'combined_final_data_zn.csv')
    EN_DATA_PATH: str = os.path.join(PROJECT_ROOT, 'data', 'data_en.csv')
    API_TITLE: str = "舆情情感分析API"
    API_VERSION: str = "4.0.0"
    AUTHOR: str = "罗广太" # <-- 作者姓名在这里定义
    AUTHOR_EMAIL: str = "3198933505@qq.com" # <-- 建议添加一个邮箱

class AppState:
    """应用全局状态，存放加载后的实例"""
    data_service: DataService
    analysis_service: AnalysisService

app_state = AppState()

# --- 4. FastAPI应用与生命周期事件 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器：在启动时加载模型和数据
    """
    logger.info("应用启动中... 正在加载模型和数据...")

    # 1. 首先创建 DataService 实例
    app_state.data_service = DataService(AppConfig.ZH_DATA_PATH, AppConfig.EN_DATA_PATH)

    # 2. 初始化其他模型
    sentiment_predictor = SentimentPredictor()
    keyword_extractor = KeywordExtractor()

    # 3. 创建 AnalysisService，并传入依赖
    app_state.analysis_service = AnalysisService(
        data_service=app_state.data_service,
        predictor=sentiment_predictor,
        extractor=keyword_extractor
    )

    logger.info("API已就绪，所有模型和数据均已预加载。")
    yield
    logger.info("应用关闭中...")

# --- 关键步骤：定义将在文档中显示的联系人信息 ---
contact_info = {
    "name": AppConfig.AUTHOR,
    "email": AppConfig.AUTHOR_EMAIL,
}

print(f"--- FastAPI 应用即将创建，版本号: {AppConfig.API_VERSION} ---")
app = FastAPI(
    title=AppConfig.API_TITLE,
    version=AppConfig.API_VERSION,
    description="一个用于舆情分析的API，支持情感分析、关键词提取和热度趋势。", # 添加描述信息
    lifespan=lifespan,
    contact=contact_info  # <--- 在这里将联系人信息传递给FastAPI
)

# --- 修正CORS配置 ---
origins = [
    "http://localhost:63342",
    "http://127.0.0.1:63342",
    "null"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 5. API 端点 ---
@app.post("/analyze", response_model=AnalysisResponse, summary="执行舆情分析")
async def analyze_topic(request: AnalysisRequest):
    """接收请求，并将其委托给 AnalysisService 进行处理。"""
    return app_state.analysis_service.run_analysis(request=request)

@app.get("/", tags=["系统"], summary="健康检查")
async def root():
    """提供一个简单的健康检查端点。"""
    return {"message": "舆情分析API已启动", "docs_url": "/docs"}


# --- 6. 应用启动 (用于直接运行调试) ---
if __name__ == "__main__":
    import uvicorn
    # 确保在终端中使用 uvicorn main:app --host 127.0.0.1 --port 8001 --reload 启动
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)