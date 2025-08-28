# 社交文本情感分析系统 (Social Text Sentiment Analysis System)

-----

这是一个为了应对**小学期实践项目**个人搭建的简单社交文本情感分析系统。它旨在通过前沿的自然语言处理技术，为中文和英文社交媒体文本提供深度、多维度的舆情分析。

系统后端采用高性能的 FastAPI 框架构建，前端则通过 ECharts 实现了功能丰富、交互友好的数据可视化看板，为用户提供从宏观趋势到微观归因的全方位洞察。

## ✨ 核心功能

  * **多语言支持**: 无缝处理 **中文** 与 **英文** 文本，满足跨语言分析需求。

  * **多模型情感分析引擎**:

      * **中文**: 集成 `BERT` 微调模型（精度高）和 `SnowNLP`（速度快），用户可按需切换。
      * **英文**: 集成 `DistilBERT` 微调模型（效果好）和 `VADER`（轻量级），适应不同场景。

  * **先进的关键词提取策略**:

      * 内置多种业界前沿的关键词提取算法，包括 `TextRank`, `KeyBERT`, `Hybrid` (混合模式), `POS-Guided KeyBERT` (词性引导)。
      * 独创 **`Hybrid Ensemble` (混合集成)** 策略，通过对多种算法结果进行加权投票，显著提升关键词提取的准确性和相关性，是项目推荐的最佳实践。

  * **全方位可视化分析看板**:

      * **情感分布饼图**: 宏观展示正面、中性、负面情感的总体比例。
      * **情感趋势时序图**: 动态追踪特定话题的情感随时间（天）的波动情况。
      * **全时段热点趋势图**: 对比分析多个话题在选定时间范围内的热度变化，发现热门话题。
      * **联动情感归因词云**:
          * **优点/缺点关键词词云**: 深入分析引发正面和负面情感的核心原因，实现精准的情感归因。
          * **综合关键词词云**: 快速把握当前话题下的整体核心焦点。

  * **企业级高性能API**:

      * 基于 `FastAPI` 构建异步、高性能的API服务。
      * 通过 `lifespan` 生命周期管理，在应用启动时 **预加载所有模型和数据**，将API响应时间降至最低，确保流畅的用户体验。
      * 精巧的 `DataService` 和 `AnalysisService` 分层设计，实现业务逻辑与数据处理的完全解耦，提升了代码的可维护性和扩展性。

## 🛠️ 技术栈

  * **后端**: Python, FastAPI
  * **核心NLP库**:
      * **情感分析**: `transformers` (Hugging Face), `torch`, `snownlp`, `nltk`
      * **关键词提取**: `keybert`, `spacy`, `pytextrank`, `jieba`
  * **数据处理**: `pandas`, `pytz`
  * **前端**: HTML, JavaScript (原生)
  * **可视化库**: `echarts`, `echarts-wordcloud`
  * **性能测试**: `locust`

## 🚀 如何运行

### 1\. 环境准备与依赖安装

首先，请确保您已安装 Python 3.8+(最好是Python 3.11.9)。然后，在项目根目录下创建一个虚拟环境并安装所有依赖。

```bash
# 创建虚拟环境 (推荐)
python -m venv venv
# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2\. 模型与数据准备

请确保 `models/` 文件夹和 `data/` 文件夹已按项目结构放置在正确的位置。系统启动时会自动加载这些资源。

  * **models/**：存储所有预训练和微调的情绪分析模型。
  * **data/**：存储用于分析的CSV表格数据文件。

### 3\. 启动后端 API 服务

在项目根目录下，进入 `api` 文件夹，然后通过 `uvicorn` 启动后端服务。

```bash
cd api
# 启动 FastAPI service
# --reload 选项可以在代码更改后自动重启服务，适用于开发环境
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
```

服务成功启动后，您将在终端中看到以下输出：
`Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)`

现在，您可以访问 `http://127.0.0.1:8001/docs` 来查看和测试自动生成的 API 文档 (Swagger UI)。

### 4\. 访问前端可视化看板

当后端服务运行时，您可以通过直接在现代浏览器（例如 Chrome、Firefox）中打开 `frontend/frontend_api_test.html` 文件来访问意见面板。

**使用步骤：**

1.  **选择语言**：从语言下拉菜单中选择中文或英文。
2.  **选择情感模型**: 根据所选语言，选择相应的分析模型。
3.  **输入话题**: 在“话题”输入框中填写您想分析的关键词（例如：“迪士尼”）。
4.  **选择日期范围**：设置要分析的时间段。
5.  **选择关键字算法**：强烈建议使用默认的“混合集成（最佳）”策略以获得最佳效果。
6.  **开始分析**：点击“开始分析”按钮，前端会向后端 API 发送请求。
7.  **查看结果**: 分析完成后，页面将动态渲染出所有的分析图表。还可以查看底部的原始 JSON 响应进行调试。

## 📁 项目结构概览

```
.
├── api/                             # 核心后端 API 代码
│   ├── core/                        # 核心算法模块（与业务逻辑解耦）
│   │   ├── keyword_extractor.py     # 封装了所有关键词提取策略
│   │   └── sentiment_predictor.py   # 封装所有情绪分析模型调用
│   ├── services/                    # 服务层（处理业务逻辑）
│   │   └── analysis_service.py      # 核心分析服务、编排算法和数据处理
│   ├── main.py                      # FastAPI 应用门户，定义 API 路由和生命周期
│   └── schemas.py                   # 用于请求和响应数据验证的 pydantic 数据模型
├── data/                            # 数据集
│   ├── combined_final_data_zn.csv   # 中文演示数据集
│   └── data_en.csv                  # 英文简报数据集
├── frontend/                        # 前端代码
│   └── frontend_api_test.html       # 单文件前端看板页面
├── models/                          # 预训练和微调模型
│   ├── bert-finetuned-sentiment/    # 微调中文 BERT 模型
│   └── distilbert-base-uncased-finetuned-sst-2-english/ # 微调英文 DistilBERT 模型
├── experiments/                     # 模型评估和实验代码（Jupyter Notebook 或 Python 脚本）
│   ├── exp_1_baseline_model.py      # 基线模型评估
│   ├── ...                          # 其他模型和算法的对比实验
│   └── exp_6_keyword_comparison_fixed.py # 关键词提取算法对比
├── reports/                         # 存储实验报告和图表
│   └── figures/                     # 存储实验生成的图表
├── config.py                        # 全局配置文件，用于定义路径和常量
├── locustfile.py                    # 用于 API 性能测试的 locust 脚本
└── all_models_performance.json      # 比较各模型性能的JSON数据
```
