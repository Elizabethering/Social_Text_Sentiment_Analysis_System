# import os
# import jieba
# import jieba.analyse
# import jieba.posseg as pseg
# import spacy
# import pytextrank
# from keybert import KeyBERT
# from sklearn.feature_extraction.text import TfidfVectorizer
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# import time
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # --- 1. Initialize Models ---
# print("Initializing models...")
# nlp_en = spacy.load("en_core_web_sm")
# if "textrank" not in nlp_en.pipe_names:
#     nlp_en.add_pipe("textrank")
#
# kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
#
#
# # --- 2. Load Data from Folders ---
# def load_texts_from_folder(folder_path):
#     """Load all text files from a folder"""
#     texts = []
#     if not os.path.exists(folder_path):
#         print(f"Warning: Folder {folder_path} does not exist!")
#         return texts
#
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.txt'):
#             file_path = os.path.join(folder_path, filename)
#             try:
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     content = f.read().strip()
#                     if content:
#                         texts.append(content)
#             except Exception as e:
#                 print(f"Error reading {file_path}: {e}")
#
#     return texts
#
#
# # --- 3. Define Keyword Extraction Functions ---
#
# # 【Strategy 0: TF-IDF (Why we eliminate it)】
# def extract_keywords_tfidf(texts, text_index, is_chinese=False, top_k=5):
#     """
#     TF-IDF requires a corpus for comparison.
#     This demonstrates why it's not ideal for single document keyword extraction.
#     """
#     if len(texts) < 2:
#         return ["Need multiple documents for TF-IDF"]
#
#     if is_chinese:
#         # Tokenize Chinese text
#         texts_tokenized = [' '.join(jieba.cut(text)) for text in texts]
#     else:
#         texts_tokenized = texts
#
#     try:
#         vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
#         tfidf_matrix = vectorizer.fit_transform(texts_tokenized)
#         feature_names = vectorizer.get_feature_names_out()
#
#         # Get scores for the specific document
#         scores = tfidf_matrix[text_index].toarray()[0]
#         top_indices = scores.argsort()[-top_k:][::-1]
#
#         return [feature_names[i] for i in top_indices if scores[i] > 0]
#     except:
#         return ["TF-IDF failed"]
#
#
# # 【Strategy 1: TextRank Only】
# def extract_keywords_textrank(text, is_chinese=False, top_k=5):
#     if is_chinese:
#         return jieba.analyse.textrank(text, topK=top_k, withWeight=False)
#     else:
#         doc = nlp_en(text.lower())
#         return [p.text for p in doc._.phrases[:top_k]]
#
#
# # 【Strategy 2: KeyBERT Only (Pure)】
# def extract_keywords_keybert_pure(text, is_chinese=False, top_n=5):
#     keyphrase_range = (1, 3)
#     stop_words_lang = 'english' if not is_chinese else None
#
#     keywords = kw_model.extract_keywords(text,
#                                          keyphrase_ngram_range=keyphrase_range,
#                                          stop_words=stop_words_lang,
#                                          top_n=top_n,
#                                          use_mmr=True,
#                                          diversity=0.5)
#     return [k for k, v in keywords]
#
#
# # 【Strategy 3: TextRank → KeyBERT (Sequential)】
# def extract_keywords_textrank_then_keybert(text, is_chinese=False, top_k=5):
#     # Step 1: Generate candidates with TextRank
#     if is_chinese:
#         tr_candidates = jieba.analyse.textrank(text, topK=20, withWeight=False)
#     else:
#         doc = nlp_en(text.lower())
#         tr_candidates = [p.text for p in doc._.phrases[:20]]
#
#     if not tr_candidates:
#         return []
#
#     # Step 2: Re-rank with KeyBERT
#     keywords = kw_model.extract_keywords(text,
#                                          candidates=tr_candidates,
#                                          top_n=top_k,
#                                          use_mmr=True,
#                                          diversity=0.3)
#     return [k for k, v in keywords]
#
#
# # 【Strategy 4: KeyBERT → TextRank (Reverse Sequential)】
# def extract_keywords_keybert_then_textrank(text, is_chinese=False, top_k=5):
#     # Step 1: Get candidates from KeyBERT
#     kb_candidates = extract_keywords_keybert_pure(text, is_chinese=is_chinese, top_n=20)
#     if not kb_candidates:
#         return []
#
#     # Step 2: Re-rank with TextRank
#     candidate_text = " ".join(kb_candidates)
#     if is_chinese:
#         return jieba.analyse.textrank(candidate_text, topK=top_k, withWeight=False)
#     else:
#         doc = nlp_en(candidate_text.lower())
#         phrases = [p.text for p in doc._.phrases[:top_k]]
#         if phrases:
#             return phrases
#         # Fallback to tokens if no phrases found
#         return [token.text for token in doc if not token.is_stop and not token.is_punct][:top_k]
#
#
# # 【Strategy 5: POS-Guided KeyBERT (Optimized)】
# def extract_keywords_pos_keybert(text, is_chinese=False, top_n=5):
#     """
#     Use POS tagging to create high-quality candidates for KeyBERT
#     """
#     candidates = []
#
#     if is_chinese:
#         # Extract nouns, verbs, and adjectives
#         word_flags = pseg.cut(text)
#         for word, flag in word_flags:
#             if len(word) > 1 and (flag.startswith('n') or flag.startswith('v') or flag.startswith('a')):
#                 candidates.append(word)
#     else:
#         # Extract nouns, proper nouns, and adjectives
#         doc = nlp_en(text)
#         candidates = []
#         # Single words
#         for token in doc:
#             if token.pos_ in ('NOUN', 'PROPN', 'ADJ') and not token.is_stop:
#                 candidates.append(token.lemma_)
#         # Noun phrases
#         for chunk in doc.noun_chunks:
#             candidates.append(chunk.text.lower())
#
#     if not candidates:
#         return extract_keywords_keybert_pure(text, is_chinese, top_n)
#
#     # Remove duplicates while preserving order
#     candidates = list(dict.fromkeys(candidates))
#
#     keywords = kw_model.extract_keywords(text,
#                                          candidates=candidates,
#                                          top_n=top_n,
#                                          use_mmr=True,
#                                          diversity=0.4)
#     return [k for k, v in keywords]
#
#
# # 【Strategy 6: Hybrid Ensemble (New Optimization)】
# def extract_keywords_hybrid_ensemble(text, is_chinese=False, top_k=5):
#     """
#     Combine multiple methods and use voting/scoring to get the best keywords
#     """
#     keyword_scores = defaultdict(float)
#
#     # Get keywords from different methods
#     textrank_kws = extract_keywords_textrank(text, is_chinese, top_k=10)
#     keybert_kws = extract_keywords_keybert_pure(text, is_chinese, top_n=10)
#     pos_keybert_kws = extract_keywords_pos_keybert(text, is_chinese, top_n=10)
#
#     # Scoring based on position and frequency
#     for i, kw in enumerate(textrank_kws):
#         keyword_scores[kw.lower()] += (10 - i) * 1.0  # TextRank weight
#
#     for i, kw in enumerate(keybert_kws):
#         keyword_scores[kw.lower()] += (10 - i) * 1.2  # KeyBERT weight (slightly higher)
#
#     for i, kw in enumerate(pos_keybert_kws):
#         keyword_scores[kw.lower()] += (10 - i) * 1.1  # POS-guided weight
#
#     # Sort by score and return top k
#     sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
#     return [kw for kw, score in sorted_keywords[:top_k]]
#
#
# # --- 4. Evaluation Metrics ---
# def evaluate_strategies(texts, is_chinese=False, sample_size=5):
#     """
#     Compare different strategies on multiple texts
#     """
#     results = {
#         'TF-IDF': [],
#         'TextRank': [],
#         'KeyBERT': [],
#         'TextRank→KeyBERT': [],
#         'KeyBERT→TextRank': [],
#         'POS-KeyBERT': [],
#         'Hybrid Ensemble': []
#     }
#
#     times = {key: [] for key in results.keys()}
#
#     # Sample texts for evaluation
#     sample_texts = texts[:min(sample_size, len(texts))]
#
#     for idx, text in enumerate(sample_texts):
#         print(f"\n{'=' * 80}")
#         print(f"Text {idx + 1} Preview: {text[:150]}...")
#         print(f"{'=' * 80}")
#
#         # Strategy 0: TF-IDF
#         start = time.time()
#         tfidf_kws = extract_keywords_tfidf(texts, idx, is_chinese)
#         times['TF-IDF'].append(time.time() - start)
#         results['TF-IDF'].append(tfidf_kws)
#         print(f"{'TF-IDF':<20}: {tfidf_kws}")
#
#         # Strategy 1: TextRank
#         start = time.time()
#         textrank_kws = extract_keywords_textrank(text, is_chinese)
#         times['TextRank'].append(time.time() - start)
#         results['TextRank'].append(textrank_kws)
#         print(f"{'TextRank':<20}: {textrank_kws}")
#
#         # Strategy 2: KeyBERT
#         start = time.time()
#         keybert_kws = extract_keywords_keybert_pure(text, is_chinese)
#         times['KeyBERT'].append(time.time() - start)
#         results['KeyBERT'].append(keybert_kws)
#         print(f"{'KeyBERT':<20}: {keybert_kws}")
#
#         # Strategy 3: TextRank → KeyBERT
#         start = time.time()
#         tr_kb_kws = extract_keywords_textrank_then_keybert(text, is_chinese)
#         times['TextRank→KeyBERT'].append(time.time() - start)
#         results['TextRank→KeyBERT'].append(tr_kb_kws)
#         print(f"{'TextRank→KeyBERT':<20}: {tr_kb_kws}")
#
#         # Strategy 4: KeyBERT → TextRank
#         start = time.time()
#         kb_tr_kws = extract_keywords_keybert_then_textrank(text, is_chinese)
#         times['KeyBERT→TextRank'].append(time.time() - start)
#         results['KeyBERT→TextRank'].append(kb_tr_kws)
#         print(f"{'KeyBERT→TextRank':<20}: {kb_tr_kws}")
#
#         # Strategy 5: POS-KeyBERT
#         start = time.time()
#         pos_kb_kws = extract_keywords_pos_keybert(text, is_chinese)
#         times['POS-KeyBERT'].append(time.time() - start)
#         results['POS-KeyBERT'].append(pos_kb_kws)
#         print(f"{'POS-KeyBERT':<20}: {pos_kb_kws}")
#
#         # Strategy 6: Hybrid Ensemble
#         start = time.time()
#         hybrid_kws = extract_keywords_hybrid_ensemble(text, is_chinese)
#         times['Hybrid Ensemble'].append(time.time() - start)
#         results['Hybrid Ensemble'].append(hybrid_kws)
#         print(f"{'Hybrid Ensemble':<20}: {hybrid_kws}")
#
#     return results, times
#
#
# # --- 5. Why TF-IDF is Eliminated ---
# def demonstrate_tfidf_problems():
#     print("\n" + "=" * 80)
#     print(" Why TF-IDF is Not Ideal for Single Document Keyword Extraction ".center(80, "="))
#     print("=" * 80)
#     print("\nProblems with TF-IDF:")
#     print("1. **Corpus Dependency**: TF-IDF requires multiple documents for comparison")
#     print("2. **Context Loss**: It treats documents as bag-of-words, losing semantic relationships")
#     print("3. **No Semantic Understanding**: Can't capture synonyms or related concepts")
#     print("4. **Poor with Short Texts**: Performance degrades with limited text")
#     print("5. **Language Specific**: Requires different preprocessing for each language")
#     print("\nExample: TF-IDF might rank common words high if they're rare in the corpus,")
#     print("even if they're not truly important to the document's meaning.")
#
#
# # --- 6. Performance Summary ---
# def print_performance_summary(times_zh, times_en):
#     print("\n" + "=" * 80)
#     print(" Performance Summary ".center(80, "="))
#     print("=" * 80)
#
#     # Calculate average times
#     all_times = {}
#     for strategy in times_zh.keys():
#         zh_avg = np.mean(times_zh[strategy]) if times_zh[strategy] else 0
#         en_avg = np.mean(times_en[strategy]) if times_en[strategy] else 0
#         all_times[strategy] = (zh_avg + en_avg) / 2
#
#     # Sort by speed
#     sorted_strategies = sorted(all_times.items(), key=lambda x: x[1])
#
#     print("\nAverage Processing Time (seconds):")
#     for strategy, avg_time in sorted_strategies:
#         print(f"  {strategy:<20}: {avg_time:.4f}s")
#
#     print("\n" + "=" * 80)
#     print(" Recommendations ".center(80, "="))
#     print("=" * 80)
#     print("\n1. **For Speed**: TextRank is fastest but less accurate")
#     print("2. **For Quality**: POS-KeyBERT provides best semantic understanding")
#     print("3. **For Balance**: TextRank→KeyBERT combines speed with quality")
#     print("4. **For Best Results**: Hybrid Ensemble, though slower, gives most comprehensive keywords")
#     print("5. **Avoid TF-IDF**: For single document analysis due to corpus dependency")
#
#
# # --- 7. Main Execution ---
# def main():
#     print("Starting Keyword Extraction Experiment...")
#
#     # Load data
#     print("\nLoading Chinese texts...")
#     texts_zh = load_texts_from_folder('data/data_zh')
#
#     print(f"Loading English texts...")
#     texts_en = load_texts_from_folder('data/data_en')
#
#     # Use sample data if folders don't exist
#     if not texts_zh:
#         print("Using sample Chinese texts...")
#         texts_zh = [
#             "这家餐厅的服务态度真的太差了，上菜慢得离谱，以后再也不会来了！",
#             "今天天气真好，阳光明媚，很适合跟朋友一起出去野餐，心情都变好了。",
#             "刚看完这部电影，剧情反转再反转，特效也超级棒，绝对是年度最佳悬疑片！",
#             "新买的这个耳机降噪效果绝了，戴上之后世界都安静了，音质也很清晰。",
#             "人工智能技术正在快速发展，深度学习和神经网络已经在图像识别、自然语言处理等领域取得了突破性进展。"
#         ]
#
#     if not texts_en:
#         print("Using sample English texts...")
#         texts_en = [
#             "The customer service at this store is absolutely terrible. Waited 20 minutes just to be ignored.",
#             "What a beautiful day! The sun is shining and it's perfect weather for a picnic with friends.",
#             "Just finished the new season of this show. The plot twists were insane and the ending left me speechless.",
#             "This gaming laptop is a beast! It runs everything on ultra settings without dropping frames.",
#             "Machine learning algorithms have revolutionized data analysis, enabling computers to learn patterns from data."
#         ]
#
#     # Demonstrate why TF-IDF is eliminated
#     demonstrate_tfidf_problems()
#
#     # Evaluate Chinese texts
#     print("\n\n" + "=" * 80)
#     print(" CHINESE TEXT ANALYSIS ".center(80, "="))
#     print("=" * 80)
#     results_zh, times_zh = evaluate_strategies(texts_zh, is_chinese=True)
#
#     # Evaluate English texts
#     print("\n\n" + "=" * 80)
#     print(" ENGLISH TEXT ANALYSIS ".center(80, "="))
#     print("=" * 80)
#     results_en, times_en = evaluate_strategies(texts_en, is_chinese=False)
#
#     # Print performance summary
#     print_performance_summary(times_zh, times_en)
#
#
# if __name__ == "__main__":
#     main()

import os
import jieba
import jieba.analyse
import jieba.posseg as pseg
import spacy
import pytextrank  # 确保已安装
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from collections import defaultdict
import time
import warnings

warnings.filterwarnings('ignore')

# --- 1. Initialize Models ---
print("Initializing models...")
nlp_en = spacy.load("en_core_web_sm")
if "textrank" not in nlp_en.pipe_names:
    nlp_en.add_pipe("textrank")

kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')


# --- 2. Load Data from Folders (函数保持不变) ---
def load_texts_from_folder(folder_path):
    # ... (代码无变化)
    texts = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist!")
        return texts
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts.append(content)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return texts


# --- 3. Define Keyword Extraction Functions (函数保持不变) ---
def extract_keywords_tfidf(texts, text_index, is_chinese=False, top_k=5):
    # ... (代码无变化)
    if len(texts) < 2:
        return ["Need multiple documents for TF-IDF"]
    if is_chinese:
        texts_tokenized = [' '.join(jieba.cut(text)) for text in texts]
    else:
        texts_tokenized = texts
    try:
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts_tokenized)
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix[text_index].toarray()[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        return [feature_names[i] for i in top_indices if scores[i] > 0]
    except:
        return ["TF-IDF failed"]


def extract_keywords_textrank(text, is_chinese=False, top_k=5):
    # ... (代码无变化)
    if is_chinese:
        return jieba.analyse.textrank(text, topK=top_k, withWeight=False)
    else:
        doc = nlp_en(text.lower())
        return [p.text for p in doc._.phrases[:top_k]]


def extract_keywords_keybert_pure(text, is_chinese=False, top_n=5):
    # ... (代码无变化)
    keyphrase_range = (1, 3)
    stop_words_lang = 'english' if not is_chinese else None
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=keyphrase_range, stop_words=stop_words_lang,
                                         top_n=top_n, use_mmr=True, diversity=0.5)
    return [k for k, v in keywords]


def extract_keywords_textrank_then_keybert(text, is_chinese=False, top_k=5):
    # ... (代码无变化)
    if is_chinese:
        tr_candidates = jieba.analyse.textrank(text, topK=20, withWeight=False)
    else:
        doc = nlp_en(text.lower())
        tr_candidates = [p.text for p in doc._.phrases[:20]]
    if not tr_candidates: return []
    keywords = kw_model.extract_keywords(text, candidates=tr_candidates, top_n=top_k, use_mmr=True, diversity=0.3)
    return [k for k, v in keywords]


def extract_keywords_keybert_then_textrank(text, is_chinese=False, top_k=5):
    # ... (代码无变化)
    kb_candidates = extract_keywords_keybert_pure(text, is_chinese=is_chinese, top_n=20)
    if not kb_candidates: return []
    candidate_text = " ".join(kb_candidates)
    if is_chinese:
        return jieba.analyse.textrank(candidate_text, topK=top_k, withWeight=False)
    else:
        doc = nlp_en(candidate_text.lower())
        phrases = [p.text for p in doc._.phrases[:top_k]]
        if phrases: return phrases
        return [token.text for token in doc if not token.is_stop and not token.is_punct][:top_k]


def extract_keywords_pos_keybert(text, is_chinese=False, top_n=5):
    # ... (代码无变化)
    candidates = []
    if is_chinese:
        word_flags = pseg.cut(text)
        for word, flag in word_flags:
            if len(word) > 1 and (flag.startswith('n') or flag.startswith('v') or flag.startswith('a')):
                candidates.append(word)
    else:
        doc = nlp_en(text)
        for token in doc:
            if token.pos_ in ('NOUN', 'PROPN', 'ADJ') and not token.is_stop:
                candidates.append(token.lemma_)
        for chunk in doc.noun_chunks:
            candidates.append(chunk.text.lower())
    if not candidates: return extract_keywords_keybert_pure(text, is_chinese, top_n)
    candidates = list(dict.fromkeys(candidates))
    keywords = kw_model.extract_keywords(text, candidates=candidates, top_n=top_n, use_mmr=True, diversity=0.4)
    return [k for k, v in keywords]


def extract_keywords_hybrid_ensemble(text, is_chinese=False, top_k=5):
    # ... (代码无变化)
    keyword_scores = defaultdict(float)
    textrank_kws = extract_keywords_textrank(text, is_chinese, top_k=10)
    keybert_kws = extract_keywords_keybert_pure(text, is_chinese, top_n=10)
    pos_keybert_kws = extract_keywords_pos_keybert(text, is_chinese, top_n=10)
    for i, kw in enumerate(textrank_kws): keyword_scores[kw.lower()] += (10 - i) * 1.0
    for i, kw in enumerate(keybert_kws): keyword_scores[kw.lower()] += (10 - i) * 1.2
    for i, kw in enumerate(pos_keybert_kws): keyword_scores[kw.lower()] += (10 - i) * 1.1
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, score in sorted_keywords[:top_k]]


# --- 4. Evaluation Metrics (函数保持不变) ---
def evaluate_strategies(texts, is_chinese=False, sample_size=5):
    # ... (代码无变化)
    results = {'TF-IDF': [], 'TextRank': [], 'KeyBERT': [], 'TextRank→KeyBERT': [], 'KeyBERT→TextRank': [],
               'POS-KeyBERT': [], 'Hybrid Ensemble': []}
    times = {key: [] for key in results.keys()}
    sample_texts = texts[:min(sample_size, len(texts))]
    for idx, text in enumerate(sample_texts):
        print(f"\n{'=' * 80}")
        print(f"Text {idx + 1} Preview: {text[:150]}...")
        print(f"{'=' * 80}")
        start = time.time();
        tfidf_kws = extract_keywords_tfidf(texts, idx, is_chinese);
        times['TF-IDF'].append(time.time() - start);
        results['TF-IDF'].append(tfidf_kws);
        print(f"{'TF-IDF':<20}: {tfidf_kws}")
        start = time.time();
        textrank_kws = extract_keywords_textrank(text, is_chinese);
        times['TextRank'].append(time.time() - start);
        results['TextRank'].append(textrank_kws);
        print(f"{'TextRank':<20}: {textrank_kws}")
        start = time.time();
        keybert_kws = extract_keywords_keybert_pure(text, is_chinese);
        times['KeyBERT'].append(time.time() - start);
        results['KeyBERT'].append(keybert_kws);
        print(f"{'KeyBERT':<20}: {keybert_kws}")
        start = time.time();
        tr_kb_kws = extract_keywords_textrank_then_keybert(text, is_chinese);
        times['TextRank→KeyBERT'].append(time.time() - start);
        results['TextRank→KeyBERT'].append(tr_kb_kws);
        print(f"{'TextRank→KeyBERT':<20}: {tr_kb_kws}")
        start = time.time();
        kb_tr_kws = extract_keywords_keybert_then_textrank(text, is_chinese);
        times['KeyBERT→TextRank'].append(time.time() - start);
        results['KeyBERT→TextRank'].append(kb_tr_kws);
        print(f"{'KeyBERT→TextRank':<20}: {kb_tr_kws}")
        start = time.time();
        pos_kb_kws = extract_keywords_pos_keybert(text, is_chinese);
        times['POS-KeyBERT'].append(time.time() - start);
        results['POS-KeyBERT'].append(pos_kb_kws);
        print(f"{'POS-KeyBERT':<20}: {pos_kb_kws}")
        start = time.time();
        hybrid_kws = extract_keywords_hybrid_ensemble(text, is_chinese);
        times['Hybrid Ensemble'].append(time.time() - start);
        results['Hybrid Ensemble'].append(hybrid_kws);
        print(f"{'Hybrid Ensemble':<20}: {hybrid_kws}")
    return results, times


# --- 5. Performance Summary (函数保持不变) ---
def print_performance_summary(times_zh, times_en):
    # ... (代码无变化)
    print("\n" + "=" * 80);
    print(" Performance Summary ".center(80, "="));
    print("=" * 80)
    all_times = {}
    for strategy in times_zh.keys():
        zh_avg = np.mean(times_zh[strategy]) if times_zh[strategy] else 0
        en_avg = np.mean(times_en[strategy]) if times_en[strategy] else 0
        all_times[strategy] = (zh_avg + en_avg) / 2
    sorted_strategies = sorted(all_times.items(), key=lambda x: x[1])
    print("\nAverage Processing Time (seconds):")
    for strategy, avg_time in sorted_strategies: print(f"  {strategy:<20}: {avg_time:.4f}s")
    print("\n" + "=" * 80);
    print(" Recommendations ".center(80, "="));
    print("=" * 80)
    print("\n1. **For Speed**: TextRank is fastest but less accurate")
    print("2. **For Quality**: POS-KeyBERT provides best semantic understanding")
    print("3. **For Balance**: TextRank→KeyBERT combines speed with quality")
    print("4. **For Best Results**: Hybrid Ensemble, though slower, gives most comprehensive keywords")
    print("5. **Avoid TF-IDF**: For single document analysis due to corpus dependency")


# --- 6. Main Execution ---
def main():
    print("Starting Keyword Extraction Experiment...")

    # --- 【修改】扩展示例文本 ---
    print("\nLoading texts...")
    # 尝试从文件夹加载，如果失败则使用下面的示例
    texts_zh = load_texts_from_folder('data/data_zh')
    texts_en = load_texts_from_folder('data/data_en')

    if not texts_zh:
        print("Folder 'data/data_zh' not found. Using enhanced sample Chinese texts...")
        texts_zh = [
            "这家餐厅的服务态度真的太差了，上菜慢得离谱，以后再也不会来了！",
            "今天发布会的新款笔记本电脑，A15芯片性能起飞，屏幕素质顶级，但续航和接口数量是硬伤。",
            "人工智能技术正在快速发展，深度学习和神经网络已经在图像识别、自然语言处理等领域取得了突破性进展。",
            "《三体》这部科幻小说的想象力太宏大了，从红岸基地到黑暗森林法则，构建了一个令人震撼的宇宙社会学体系。",
            "城市内涝问题在雨季愈发严重，暴露了我们排水系统建设的短板，需要系统性的规划和长期的投入来解决。"
        ]

    if not texts_en:
        print("Folder 'data/data_en' not found. Using enhanced sample English texts...")
        texts_en = [
            "The customer service at this airline is absolutely terrible. My flight was canceled without proper notification and their staff was unhelpful.",
            "Just finished the new season of the show. The plot twists were insane, the character development was superb, and the ending left me speechless.",
            "This gaming laptop is a beast! It runs everything on ultra settings without dropping frames, but the battery life is a significant drawback.",
            "Machine learning algorithms have revolutionized data analysis, enabling computers to learn complex patterns from data without being explicitly programmed.",
            "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune, Or to take Arms against a Sea of troubles, And by opposing end them: to die, to sleep."
        ]

    # --- 【修改】移除 TF-IDF 的独立演示，它会在 evaluate_strategies 中被调用 ---

    # 评估中文文本
    print("\n\n" + "=" * 80)
    print(" CHINESE TEXT ANALYSIS ".center(80, "="))
    print("=" * 80)
    results_zh, times_zh = evaluate_strategies(texts_zh, is_chinese=True, sample_size=len(texts_zh))

    # 评估英文文本
    print("\n\n" + "=" * 80)
    print(" ENGLISH TEXT ANALYSIS ".center(80, "="))
    print("=" * 80)
    results_en, times_en = evaluate_strategies(texts_en, is_chinese=False, sample_size=len(texts_en))

    # 打印性能总结
    print_performance_summary(times_zh, times_en)


# 程序入口
if __name__ == "__main__":
    main()
