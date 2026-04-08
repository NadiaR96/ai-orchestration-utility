# metrics/metrics_tracker.py
import time
from bert_score import score as bert_score
from nltk.translate.meteor_score import single_meteor_score
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import torch
import numpy as np
import psutil

class MetricsTracker:
    def __init__(self):
        self.rouge = Rouge()

    # Performance
    def track_latency(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        latency = end - start
        return result, latency

    def memory_usage(self):
        return psutil.virtual_memory().used / (1024**2)  # MB

    # Quality / Accuracy
    def bert_score(self, candidate, reference):
        P, R, F1 = bert_score([candidate], [reference], lang="en", rescale_with_baseline=True)
        return F1.item()

    def meteor(self, candidate, reference):
        return single_meteor_score(reference, candidate)

    def rouge(self, candidate, reference):
        scores = self.rouge.get_scores(candidate, reference)
        return scores[0]

    def bleu(self, candidate, reference):
        return sentence_bleu([reference.split()], candidate.split())

    def perplexity(self, logits):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        return torch.exp(-log_probs.mean()).item()

    def hallucination_rate(self, candidate, reference_keywords):
        candidate_tokens = set(candidate.lower().split())
        extra_tokens = candidate_tokens - set(reference_keywords)
        return len(extra_tokens) / max(1, len(candidate_tokens))

    # Similarity / Embedding
    def cosine_similarity(self, vec_a, vec_b):
        a, b = np.array(vec_a), np.array(vec_b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    # Cost Tracking
    def estimate_token_cost(self, num_tokens, cost_per_1k=0.002):
        """Estimate cost for LLM tokens (default $0.002 per 1k tokens)"""
        return (num_tokens / 1000) * cost_per_1k