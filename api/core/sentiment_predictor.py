from snownlp import SnowNLP
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path


class SentimentPredictor:
    """
    一个统一的多语言情感分析预测器。
    """

    def __init__(self):
        print("Initializing Multi-lingual SentimentPredictor...")
        # 假设此文件在 api/core/ 目录下，向上两级找到项目根目录
        project_root = Path(__file__).parent.parent.parent

        # --- 1. 加载中文模型 ---
        zh_bert_model_path = project_root / 'models' / 'bert-finetuned-sentiment'
        self.zh_bert_model = None
        self.zh_bert_tokenizer = None
        self.zh_bert_loaded = False
        try:
            if zh_bert_model_path.exists():
                print(f"Loading fine-tuned Chinese BERT model from: {zh_bert_model_path}")
                self.zh_bert_model = AutoModelForSequenceClassification.from_pretrained(zh_bert_model_path)
                self.zh_bert_tokenizer = AutoTokenizer.from_pretrained(zh_bert_model_path)
                self.zh_bert_model.eval()
                self.zh_bert_loaded = True
                print("Chinese BERT model loaded successfully.")
            else:
                print(f"--- 警告: 未找到中文BERT模型路径 {zh_bert_model_path}，该模型将不可用。 ---")
        except Exception as e:
            print(f"--- 错误: 加载中文BERT模型时出错: {e} ---")

        # --- 2. 加载英文模型 ---
        # 注意: 根据您的截图，模型文件夹名为 distilbert-base-uncased-finetuned-sst-2-english
        # 我们优先从本地加载，如果本地不存在，再考虑从Hugging Face Hub下载
        en_bert_model_path = project_root / 'models' / 'distilbert-base-uncased-finetuned-sst-2-english'
        self.en_bert_model = None
        self.en_bert_tokenizer = None
        self.en_bert_loaded = False
        try:
            if en_bert_model_path.exists():
                print(f"Loading English DistilBERT model from local path: {en_bert_model_path}")
                self.en_bert_model = AutoModelForSequenceClassification.from_pretrained(en_bert_model_path)
                self.en_bert_tokenizer = AutoTokenizer.from_pretrained(en_bert_model_path)
                self.en_bert_model.eval()
                self.en_bert_loaded = True
                print("English DistilBERT model loaded successfully.")
            else:
                print(f"--- 警告: 未找到英文BERT模型路径 {en_bert_model_path}，该模型将不可用。 ---")
        except Exception as e:
            print(f"--- 错误: 加载英文BERT模型时出错: {e} ---")

        # --- 3. 初始化 VADER (作为备用) ---
        print("Initializing VADER sentiment analyzer...")
        self.vader_analyzer = SentimentIntensityAnalyzer()
        print("VADER initialized.")

        # --- 4. 标签映射 ---
        self.zh_inverse_label_map = {0: -1, 1: 0, 2: 1}  # 中文BERT: 0:负, 1:中, 2:正
        self.en_inverse_label_map = {'NEGATIVE': -1, 'POSITIVE': 1}  # 英文模型标签通常是字符串

    def predict(self, text: str, model: str, lang: str) -> int:
        """
        根据语言和指定的模型进行情感预测。

        Args:
            text (str): 需要分析的文本。
            model (str): 调用的模型 ('bert', 'snownlp')。
            lang (str): 语言 ('zh', 'en')。

        Returns:
            int: 情感标签 (-1: 负面, 0: 中性, 1: 正面)。
        """
        if lang == 'zh':
            return self._predict_zh(text, model)
        elif lang == 'en':
            # 英文目前只支持bert
            return self._predict_en(text, 'bert')
        else:
            # 不支持的语言，返回中性
            return 0

    def _predict_zh(self, text: str, model: str) -> int:
        # 明确根据 'model' 参数选择
        if model == 'bert':
            if not self.zh_bert_loaded:
                print("--- 警告: 中文BERT模型未加载，自动切换到SnowNLP。 ---")
                return self._predict_zh(text, 'snownlp')  # 如果BERT不可用，自动降级

            inputs = self.zh_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                logits = self.zh_bert_model(**inputs).logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            return self.zh_inverse_label_map.get(predicted_class_id, 0)

        # 默认或明确指定时使用SnowNLP
        elif model == 'snownlp':
            score = SnowNLP(text).sentiments
            if score > 0.6:
                return 1
            elif score < 0.4:
                return -1
            else:
                return 0
        else:
            print(f"--- 警告: 不支持的中文模型 '{model}'，使用默认的SnowNLP。 ---")
            return self._predict_zh(text, 'snownlp')

    def _predict_en(self, text: str, model: str) -> int:
        if model == 'bert':
            if not self.en_bert_loaded:
                print("--- 警告: 英文BERT模型未加载，自动切换到VADER。 ---")
                return self._predict_en(text, 'vader')  # 如果BERT不可用，自动降级

            inputs = self.en_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                logits = self.en_bert_model(**inputs).logits

            # 从logits获取概率最高的标签ID，然后通过id2label配置找到对应的标签名
            predicted_class_id = logits.argmax().item()
            label = self.en_bert_model.config.id2label[predicted_class_id]

            # 注意：此模型默认不输出中性，我们直接映射 正/负
            return self.en_inverse_label_map.get(label, 0)

        # 默认或备用使用VADER
        else:
            compound_score = self.vader_analyzer.polarity_scores(text)['compound']
            if compound_score >= 0.05:
                return 1
            elif compound_score <= -0.05:
                return -1
            else:
                return 0