import nltk
from collections import Counter
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
import psutil
import time
import math

def ensure_nltk_resource(resource_name):
    if resource_name == 'tokenizers/punkt_tab/english':
        package_name = 'punkt_tab'
    elif resource_name.startswith('tokenizers/punkt'):
        package_name = 'punkt'
    else:
        package_name = resource_name.split('/')[-1]

    try:
        nltk.data.find(resource_name)
        return
    except LookupError:
        nltk.download(package_name, quiet=True)

    # Some NLTK packages are stored as zipped resources, so try alternate paths.
    alternate_paths = [resource_name]
    if resource_name == 'corpora/wordnet':
        alternate_paths.append('corpora/wordnet.zip')
    if resource_name == 'corpora/omw-1.4':
        alternate_paths.append('corpora/omw-1.4.zip')

    for path in alternate_paths:
        try:
            nltk.data.find(path)
            return
        except LookupError:
            continue

    # If no valid resource is found, raise the original error.
    nltk.data.find(resource_name)

class MetricsTracker:
    def __init__(self):
        ensure_nltk_resource('tokenizers/punkt')
        ensure_nltk_resource('tokenizers/punkt_tab/english')
        ensure_nltk_resource('corpora/wordnet')
        ensure_nltk_resource('corpora/omw-1.4')
        self._rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    # Standard metrics
    def meteor(self, candidate, reference):
        candidate_tokens = nltk.word_tokenize(candidate)
        reference_tokens = nltk.word_tokenize(reference)
        return single_meteor_score(reference_tokens, candidate_tokens)

    def rouge(self, candidate, reference):
        scores = self._rouge_scorer.score(reference, candidate)
        return {k: v.fmeasure for k, v in scores.items()}

    def bert_score(self, candidates, references, model_type='bert-base-uncased'):
        P, R, F1 = bert_score_fn(candidates, references, lang='en', model_type=model_type)
        return float(F1.mean().item())

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

    def bleu(self, candidate, reference):
        candidate_tokens = nltk.word_tokenize(candidate)
        reference_tokens = nltk.word_tokenize(reference)
        smoothie = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

    def f1_precision_recall(self, reference_tokens, candidate_tokens):
        ref_counts = Counter(reference_tokens)
        cand_counts = Counter(candidate_tokens)
        overlap = sum(min(ref_counts[token], cand_counts[token]) for token in ref_counts)
        precision = overlap / max(1, sum(cand_counts.values()))
        recall = overlap / max(1, sum(ref_counts.values()))
        if precision + recall == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        f1 = 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1}

    def coverage_score(self, candidate, reference_tokens):
        candidate_tokens = set(nltk.word_tokenize(candidate))
        reference_set = set(reference_tokens)
        if not reference_set:
            return 0.0
        return len(candidate_tokens & reference_set) / len(reference_set)

    def latency(self, start_time, end_time):
        return end_time - start_time

    def track_latency(self, func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time

    def memory_usage(self):
        try:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def estimate_token_cost(self, num_tokens, cost_per_token=0.00001):
        return num_tokens * cost_per_token

    def cost_estimate(self, tokens_processed, cost_per_token=0.00001):
        return tokens_processed * cost_per_token