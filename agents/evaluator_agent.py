import nltk
from metrics.metrics_tracker import MetricsTracker
import time

class EvaluatorAgent:
    def __init__(self):
        self.metrics = MetricsTracker()

    def evaluate(self, candidate, reference, start_time=None, end_time=None):
        ref_tokens = nltk.word_tokenize(reference)
        results = {
            "METEOR": self.metrics.meteor(candidate, reference),
            "ROUGE": self.metrics.rouge(candidate, reference),
            "BERTScore": self.metrics.bert_score([candidate], [reference]),
            "Perplexity": self.metrics.perplexity(candidate, reference),
            "HallucinationRate": self.metrics.hallucination_rate(candidate, ref_tokens),
            "Diversity": self.metrics.diversity_score(candidate),
        }
        if start_time and end_time:
            results["Latency"] = self.metrics.latency(start_time, end_time)
        results["Cost"] = self.metrics.cost_estimate(len(candidate.split()))
        return results