import nltk
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import time
import math

# Ensure punkt tokenizer is available
nltk.download('punkt', quiet=True)

class MetricsTracker:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    # Standard metrics
    def meteor(self, candidate, reference):
        candidate_tokens = nltk.word_tokenize(candidate)
        reference_tokens = nltk.word_tokenize(reference)
        return single_meteor_score(reference_tokens, candidate_tokens)

    def rouge(self, candidate, reference):
        return self.rouge.score(reference, candidate)

    def bert_score(self, candidates, references, model_type='bert-base-uncased'):
        P, R, F1 = bert_score(candidates, references, lang='en', model_type=model_type)
        return F1.mean().item()

    def perplexity(self, candidate, reference):
        # Simple pseudo-perplexity
        candidate_tokens = nltk.word_tokenize(candidate)
        ref_tokens = set(nltk.word_tokenize(reference))
        prob = sum(1 for t in candidate_tokens if t in ref_tokens) / len(candidate_tokens)
        return math.exp(-math.log(prob+1e-8))

    # Custom metrics
    def hallucination_rate(self, candidate, reference_tokens):
        candidate_tokens = set(nltk.word_tokenize(candidate))
        hallucinated = candidate_tokens - set(reference_tokens)
        return len(hallucinated) / max(1, len(candidate_tokens))

    def diversity_score(self, text):
        tokens = nltk.word_tokenize(text)
        return len(set(tokens)) / max(1, len(tokens))

    def latency(self, start_time, end_time):
        return end_time - start_time

    def cost_estimate(self, tokens_processed, cost_per_token=0.00001):
        return tokens_processed * cost_per_token