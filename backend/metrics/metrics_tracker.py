import os
import re

bert_score_fn = None
_bert_score_import_attempted = False


def _tokenize(text):
    return re.findall(r"\b\w+\b", (text or "").lower())


def _lcs_length(left, right):
    if not left or not right:
        return 0

    rows = len(left) + 1
    cols = len(right) + 1
    dp = [[0] * cols for _ in range(rows)]

    for row in range(1, rows):
        for col in range(1, cols):
            if left[row - 1] == right[col - 1]:
                dp[row][col] = dp[row - 1][col - 1] + 1
            else:
                dp[row][col] = max(dp[row - 1][col], dp[row][col - 1])

    return dp[-1][-1]


def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except TypeError:
            return float(value.__class__.__dict__["item"]())
    return 0.0


def _coerce_bert_score(value):
    if isinstance(value, tuple) and len(value) >= 3:
        value = value[2]

    mean_fn = getattr(value, "mean", None)
    if callable(mean_fn):
        try:
            value = mean_fn()
        except TypeError:
            value = value.__class__.__dict__["mean"]()

    return _to_float(value)


def bert_score(output, reference):
    global bert_score_fn, _bert_score_import_attempted

    if not reference:
        return 0.0

    enable_bert_score = os.getenv("ENABLE_BERT_SCORE", "0") == "1"
    if enable_bert_score and not _bert_score_import_attempted and bert_score_fn is None:
        _bert_score_import_attempted = True
        try:
            from bert_score import score as _bert_score_fn

            bert_score_fn = _bert_score_fn
        except Exception:
            bert_score_fn = None

    if bert_score_fn is not None:
        try:
            return bert_score_fn([output], [reference], lang="en")
        except Exception:
            return 0.0

    output_tokens = set(_tokenize(output))
    reference_tokens = set(_tokenize(reference))
    overlap = len(output_tokens & reference_tokens)
    return overlap / max(len(reference_tokens), 1)


class MetricsTracker:
    def __init__(self):
        # Enable via: TEST_MODE=1
        self.test_mode = os.getenv("TEST_MODE", "0") == "1"

    # =====================================================
    # MAIN ENTRY POINT
    # =====================================================
    def compute_all(self, output, reference, chunks):

        # -------------------------
        # TEST MODE (CI SAFE)
        # -------------------------
        if self.test_mode:
            return {
                "bert_score": 0.5,
                "bleu": 0.5,
                "rouge": 0.5,
                "perplexity": 1.0,
                "hallucination": 0.1,
                "faithfulness": 0.5,
                "diversity": 0.5,
            }

        # -------------------------
        # NORMAL MODE
        # -------------------------
        results = {}
        references = self._normalize_references(reference)

        # Reference-based metrics
        if references:
            if len(references) == 1:
                ref_metrics = self._compute_reference_metrics(output, references[0])
            else:
                candidates = [self._compute_reference_metrics(output, ref) for ref in references]
                ref_metrics = max(candidates, key=self._reference_candidate_rank)

            results["bert_score"] = ref_metrics["bert_score"]
            results["bleu"] = ref_metrics["bleu"]
            results["rouge"] = ref_metrics["rouge"]
            results["perplexity"] = ref_metrics["perplexity"]
            results["hallucination"] = ref_metrics["hallucination"]
            results["multi_reference_count"] = float(len(references))
        else:
            results["bert_score"] = 0.0
            results["bleu"] = 0.0
            results["rouge"] = 0.0
            results["perplexity"] = 0.0
            results["hallucination"] = 0.0
            results["multi_reference_count"] = 0.0

        # Always-on metrics
        results["faithfulness"] = self.faithfulness(output, chunks)
        results["diversity"] = self.diversity_score(output)

        return results

    def _normalize_references(self, reference):
        if reference is None:
            return []

        if isinstance(reference, str):
            ref = reference.strip()
            return [ref] if ref else []

        if isinstance(reference, list):
            refs = [str(r).strip() for r in reference if str(r).strip()]
            return refs

        ref = str(reference).strip()
        return [ref] if ref else []

    def _compute_reference_metrics(self, output, reference):
        return {
            "bert_score": _coerce_bert_score(bert_score(output, reference)),
            "bleu": self.bleu(output, reference),
            "rouge": self.rouge(output, reference),
            "perplexity": self.perplexity(output, reference),
            "hallucination": self.hallucination_rate(output, _tokenize(reference)),
        }

    def _reference_candidate_rank(self, metrics):
        rouge = metrics.get("rouge") or {}
        rouge1 = rouge.get("rouge1", 0.0) if isinstance(rouge, dict) else 0.0
        return (
            (0.5 * float(metrics.get("bert_score", 0.0)))
            + (0.2 * float(metrics.get("bleu", 0.0)))
            + (0.2 * float(rouge1))
            - (0.1 * float(metrics.get("hallucination", 0.0)))
        )

    # =====================================================
    # SAFE METRIC IMPLEMENTATIONS (CI FRIENDLY)
    # =====================================================

    def bert_score(self, output, reference):
        return _coerce_bert_score(bert_score(output, reference))

    def bleu(self, output, reference):
        ref_words = _tokenize(reference)
        out_words = _tokenize(output)

        if not ref_words:
            return 0.0

        overlap = sum(1 for w in out_words if w in ref_words)
        return overlap / max(len(out_words), 1)

    def rouge(self, output, reference):
        out_tokens = _tokenize(output)
        ref_tokens = _tokenize(reference)

        out = set(out_tokens)
        ref = set(ref_tokens)

        overlap = len(out & ref)
        total = len(ref)

        return {
            "rouge1": overlap / max(total, 1),
            "rougeL": _lcs_length(out_tokens, ref_tokens) / max(len(ref_tokens), 1),
        }

    def perplexity(self, output, reference=None):
        if not output:
            return 0.0

        if not reference:
            return 0.0

        output_tokens = set(_tokenize(output))
        reference_tokens = set(_tokenize(reference))
        if not output_tokens:
            return 0.0

        overlap_ratio = len(output_tokens & reference_tokens) / len(output_tokens)
        if overlap_ratio <= 0.0:
            return 50.0

        return min(50.0, max(1.0, 1.0 / overlap_ratio))

    def hallucination_rate(self, output, reference_tokens):
        output_tokens = set(_tokenize(output))
        reference_token_set = set(_tokenize(" ".join(reference_tokens))) if reference_tokens and isinstance(reference_tokens[0], str) and len(reference_tokens) > 1 and " " in reference_tokens[0] else set(token.lower() for token in reference_tokens)

        if not output_tokens:
            return 0.0

        hallucinated = [token for token in output_tokens if token not in reference_token_set]

        return len(hallucinated) / max(len(output_tokens), 1)

    def faithfulness(self, output, chunks):
        if not chunks:
            return 0.0

        chunk_texts = [getattr(c, "text", str(c)) for c in chunks]
        context = set(_tokenize(" ".join(chunk_texts)))
        output_tokens = set(_tokenize(output))

        if not output_tokens:
            return 0.0

        overlap = sum(1 for token in output_tokens if token in context)

        return overlap / max(len(output_tokens), 1)

    def diversity_score(self, output):
        tokens = _tokenize(output)
        if not tokens:
            return 0.0

        return len(set(tokens)) / len(tokens)

    def meteor(self, output, reference):
        out_tokens = _tokenize(output)
        ref_tokens = _tokenize(reference)

        if not out_tokens or not ref_tokens:
            return 0.0

        out_set = set(out_tokens)
        ref_set = set(ref_tokens)

        matches = len(out_set & ref_set)
        if matches == 0:
            return 0.0

        precision = matches / len(out_tokens)
        recall = matches / len(ref_tokens)

        # Weighted harmonic mean (alpha=0.9 as in standard METEOR)
        # denominator is always > 0 here because matches > 0 implies both
        # precision and recall are > 0; guard is kept for defensive clarity.
        alpha = 0.9
        denominator = alpha * precision + (1 - alpha) * recall
        if denominator == 0.0:
            return 0.0
        return precision * recall / denominator

    def f1_precision_recall(self, output, reference_tokens):
        out_tokens = set(_tokenize(output))
        ref_tokens = set(token.lower() for token in reference_tokens)

        if not out_tokens or not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        overlap = len(out_tokens & ref_tokens)
        precision = overlap / len(out_tokens)
        recall = overlap / len(ref_tokens)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    def coverage_score(self, output, reference_tokens):
        out_tokens = set(_tokenize(output))
        ref_tokens = set(token.lower() for token in reference_tokens)

        if not ref_tokens:
            return 0.0

        covered = len(ref_tokens & out_tokens)
        return covered / len(ref_tokens)