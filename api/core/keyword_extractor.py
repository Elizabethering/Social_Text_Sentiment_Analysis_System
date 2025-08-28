# api/core/keyword_extractor.py
import random
from typing import List, Dict, Any
import jieba.analyse
import jieba.posseg as pseg  # 导入词性标注模块
import spacy
import pytextrank
from keybert import KeyBERT
from collections import defaultdict


class KeywordExtractor:
    """
    一个先进的、支持多策略、多语言的关键词提取器。
    在初始化时加载所有模型，以提高API调用效率。

    支持的策略 (method):
    - 'textrank':
    - 'keybert'
    - 'hybrid': KeyBERT -> TextRank
    - 'pos_guided_keybert'
    - 'hybrid_ensemble'
    """

    def __init__(self):
        print("Initializing Advanced KeywordExtractor with multiple strategies...")

        try:
            self.kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
            print("KeyBERT model loaded successfully.")
        except Exception as e:
            print(f"--- 错误: KeyBERT模型加载失败: {e} ---")
            self.kw_model = None
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            if "textrank" not in self.nlp_en.pipe_names:
                self.nlp_en.add_pipe("textrank")
            print("English spaCy model with pytextrank loaded successfully.")
        except Exception as e:
            print(f"--- 警告: spaCy 'en_core_web_sm' 模型加载失败。英文TextRank将不可用: {e} ---")
            self.nlp_en = None
        print("KeywordExtractor initialization complete.")

    def extract(self, texts: List[str], lang: str, top_k: int = 50, method: str = 'textrank') -> List[Dict[str, Any]]:
        if not texts:
            return []
        SAMPLE_SIZE = 500
        if len(texts) > SAMPLE_SIZE:
            print(
                f"--- WARNING: Input text list is too large ({len(texts)} items). Taking a random sample of {SAMPLE_SIZE} to prevent memory errors. ---")
            texts_for_keywords = random.sample(texts, SAMPLE_SIZE)
        else:
            texts_for_keywords = texts

        full_text = "\n".join(texts_for_keywords)
        # --- END OF NEW LOGIC ---

        if not full_text.strip():
            return []

        # The rest of the function remains the same, but now operates on the sampled 'full_text'
        if method == 'hybrid_ensemble':
            if self.kw_model is None or (lang == 'en' and self.nlp_en is None):
                print("--- 警告: 'hybrid_ensemble' 策略所需模型未完全加载。 ---")
                return []
            return self._extract_hybrid_ensemble(full_text, lang, top_k)

        full_text = "\n".join(texts)
        if not full_text.strip():
            return []

        # --- 核心修改：增加对新策略的调用 ---
        if method == 'hybrid_ensemble':
            if self.kw_model is None or (lang == 'en' and self.nlp_en is None):
                print("--- 警告: 'hybrid_ensemble' 策略所需模型未完全加载。 ---")
                return []
            return self._extract_hybrid_ensemble(full_text, lang, top_k)

        elif method == 'pos_guided_keybert':
            if self.kw_model is None:
                print("--- 警告: KeyBERT 模型未加载，无法使用 'pos_guided_keybert' 策略。 ---")
                return []
            return self._extract_pos_guided_keybert(full_text, lang, top_k)

        elif method == 'keybert':
            # ... (其他策略调用保持不变)
            if self.kw_model is None:
                print("--- 警告: KeyBERT 模型未加载，无法使用 'keybert' 策略。 ---")
                return []
            return self._extract_keybert(full_text, top_k)
        elif method == 'hybrid':
            if self.kw_model is None or (lang == 'en' and self.nlp_en is None):
                print("--- 警告: 'hybrid' 策略所需模型未完全加载。 ---")
                return []
            return self._extract_hybrid(full_text, lang, top_k)
        else:  # 默认 textrank
            if lang == 'en' and self.nlp_en is None:
                print("--- 警告: 英文 spaCy 模型未加载，无法对英文使用 'textrank' 策略。 ---")
                return []
            return self._extract_textrank(full_text, lang, top_k)

    # --- 新增方法：策略6 (POS-Guided KeyBERT) ---
    def _extract_pos_guided_keybert(self, text: str, lang: str, top_k: int) -> List[Dict[str, Any]]:
        """使用词性标注引导的KeyBERT提取关键词"""
        print(f"Executing POS-Guided KeyBERT for lang='{lang}'...")
        candidate_words = []
        if lang == 'zh':
            # 筛选中文名词(n)、动词(v)、形容词(a)
            allowed_pos = ('n', 'v', 'a')
            words = pseg.cut(text)
            candidate_words = [word for word, flag in words if flag.startswith(allowed_pos)]
        else:  # lang == 'en'
            # 筛选英文名词(NOUN)、专有名词(PROPN)、动词(VERB)、形容词(ADJ)
            allowed_pos = ('NOUN', 'PROPN', 'VERB', 'ADJ')
            doc = self.nlp_en(text)
            candidate_words = [token.text for token in doc if token.pos_ in allowed_pos]

        if not candidate_words:
            return []

        # 使用高质量的候选词库来引导KeyBERT
        keywords = self.kw_model.extract_keywords(
            text,
            candidates=candidate_words,
            top_n=top_k,
        )
        return [{"word": word, "weight": round(weight, 4)} for word, weight in keywords]

    # --- 新增方法：策略7 (Hybrid Ensemble) ---
    def _extract_hybrid_ensemble(self, text: str, lang: str, top_k: int) -> List[Dict[str, Any]]:
        """混合集成策略：对多种算法结果进行加权投票"""
        print("Executing Hybrid Ensemble Strategy 7...")

        # 1. 定义不同算法的权重，可根据经验调整
        weights = {
            'textrank': 0.8,
            'keybert': 1.2,
            'hybrid': 1.0,  # KeyBERT -> TextRank
            'pos_guided_keybert': 1.5  # 给予最高权重，因为它通常最准确
        }

        # 2. 并行运行所有策略
        # 为避免重复计算，top_k可以设置得稍大一些
        k_for_sub_models = top_k + 20

        textrank_results = self._extract_textrank(text, lang, k_for_sub_models)
        keybert_results = self._extract_keybert(text, k_for_sub_models)
        hybrid_results = self._extract_hybrid(text, lang, k_for_sub_models)
        pos_guided_results = self._extract_pos_guided_keybert(text, lang, k_for_sub_models)

        # 3. 加权投票计分
        ensemble_scores = defaultdict(float)

        all_results = {
            'textrank': textrank_results,
            'keybert': keybert_results,
            'hybrid': hybrid_results,
            'pos_guided_keybert': pos_guided_results
        }

        for method, results in all_results.items():
            for item in results:
                word = item['word']
                score = item['weight']
                # 累加加权分数：算法权重 * 算法内部得分
                ensemble_scores[word] += weights[method] * score

        # 4. 排序并选出最终结果
        sorted_keywords = sorted(ensemble_scores.items(), key=lambda x: x[1], reverse=True)

        # 格式化输出
        final_keywords = [
            {"word": word, "weight": round(score, 4)}
            for word, score in sorted_keywords[:top_k]
        ]

        return final_keywords

    # --- 以下是之前已有的方法，保持不变 ---
    def _extract_textrank(self, text: str, lang: str, top_k: int) -> List[Dict[str, Any]]:
        # ... (代码不变)
        print(f"Executing TextRank for lang='{lang}'...")
        if lang == 'zh':
            keywords = jieba.analyse.textrank(text, topK=top_k, withWeight=True)
        else:
            doc = self.nlp_en(text)
            keywords = [(p.text, p.rank) for p in doc._.phrases[:top_k]]
        return [{"word": word, "weight": round(weight, 4)} for word, weight in keywords]

    def _extract_keybert(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        # ... (代码不变)
        print("Executing KeyBERT...")
        keywords = self.kw_model.extract_keywords(
            text, keyphrase_ngram_range=(1, 2), stop_words=None,
            top_n=top_k, use_mmr=False, diversity=0.7
        )
        return [{"word": word, "weight": round(weight, 4)} for word, weight in keywords]

    def _extract_hybrid(self, text: str, lang: str, top_k: int) -> List[Dict[str, Any]]:
        # ... (代码不变)
        print("Executing Hybrid (KeyBERT + TextRank) method...")
        key_phrases = self.kw_model.extract_keywords(
            text, top_n=10, use_mmr=True, diversity=0.9
        )
        refined_text = "\n".join([phrase for phrase, score in key_phrases])
        if not refined_text.strip():
            return self._extract_textrank(text, lang, top_k)
        return self._extract_textrank(refined_text, lang, top_k)