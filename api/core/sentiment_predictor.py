# /core/sentiment_predictor.py

from typing import Dict

def get_sentiment_from_model(text: str, language: str) -> Dict:
    """
    一个统一的模型调用函数。
    它负责根据语言选择并调用相应的模型，返回情感分析结果。
    这是与模型预测最直接相关的核心逻辑。
    """
    print(f"核心逻辑层: 正在调用 {language} 模型分析: {text[:20]}...")
    if language == "chinese":
        # 未来这里会加载中文模型
        if "开心" in text or "满意" in text:
            return {"sentiment": "positive", "score": 0.95}
        elif "失望" in text or "垃圾" in text:
            return {"sentiment": "negative", "score": 0.89}
        else:
            return {"sentiment": "neutral", "score": 0.6}
    elif language == "english":
        # 未来这里会加载英文模型
        if "happy" in text.lower() or "love" in text.lower():
            return {"sentiment": "positive", "score": 0.98}
        elif "disappointed" in text.lower() or "bad" in text.lower():
            return {"sentiment": "negative", "score": 0.92}
        else:
            return {"sentiment": "neutral", "score": 0.55}
    return {}