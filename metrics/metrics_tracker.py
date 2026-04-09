# metrics/metrics_tracker.py
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Ensure tokenizer is downloaded
nltk.download("punkt", quiet=True)

class MetricsTracker:
    def __init__(self):
        self.rouge = Rouge()
        self.vectorizer = TfidfVectorizer()

    # -----------------------------
    # BLEU Score
    # -----------------------------
    def bleu(self, candidate, reference):
        reference_tokens = nltk.word_tokenize(reference)
        candidate_tokens = nltk.word_tokenize(candidate)
        return sentence_bleu([reference_tokens], candidate_tokens)

    # -----------------------------
    # METEOR Score
    # -----------------------------
    def meteor(self, candidate, reference):
        ref_tokens = nltk.word_tokenize(reference)
        cand_tokens = nltk.word_tokenize(candidate)
        return single_meteor_score(ref_tokens, cand_tokens)

    # -----------------------------
    # ROUGE Score
    # -----------------------------
    def rouge_score(self, candidate, reference):
        return self.rouge.get_scores(candidate, reference)[0]

    # -----------------------------
    # Cosine Similarity
    # -----------------------------
    def cosine_similarity_score(self, candidate, reference):
        tfidf = self.vectorizer.fit_transform([candidate, reference])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    # -----------------------------
    # Diversity Score (unique tokens / total tokens)
    # -----------------------------
    def diversity_score(self, text):
        tokens = nltk.word_tokenize(text)
        return len(set(tokens)) / max(len(tokens), 1)

    # -----------------------------
    # Coverage Score (overlap with reference)
    # -----------------------------
    def coverage_score(self, candidate, reference_tokens):
        candidate_tokens = set(nltk.word_tokenize(candidate))
        reference_tokens = set(reference_tokens)
        return len(candidate_tokens & reference_tokens) / max(len(reference_tokens), 1)

    # -----------------------------
    # Hallucination Rate
    # -----------------------------
    def hallucination_rate(self, candidate, reference_tokens):
        candidate_tokens = set(nltk.word_tokenize(candidate))
        reference_tokens = set(reference_tokens)
        hallucinated = candidate_tokens - reference_tokens
        return len(hallucinated) / max(len(candidate_tokens), 1)

    # -----------------------------
    # F1, Precision, Recall
    # -----------------------------
    def f1_precision_recall(self, reference_tokens, candidate_tokens):
        ref_set = set(reference_tokens)
        cand_set = set(candidate_tokens)
        tp = len(ref_set & cand_set)
        precision = tp / max(len(cand_set), 1)
        recall = tp / max(len(ref_set), 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
        return {"F1": f1, "Precision": precision, "Recall": recall}