class MetricRegistry:
    def __init__(self, tracker):
        self.tracker = tracker

        self.registry = {
            "meteor": self._safe(self._meteor_wrapper),
            "rouge": self._safe(self.tracker.rouge),
            "bert_score": self._safe(self.tracker.bert_score),
            "perplexity": self._safe(self.tracker.perplexity),
            "hallucination": self._safe(self._hallucination_wrapper),
            "diversity": self._safe(self.tracker.diversity_score),
            "bleu": self._safe(self.tracker.bleu),
            "f1": self._safe(self._f1_wrapper),
            "coverage": self._safe(self._coverage_wrapper),
        }

    def _safe(self, func):
        def wrapper(candidate, reference):
            try:
                return func(candidate, reference)
            except Exception as e:
                return {"error": str(e)}
        return wrapper

    def _meteor_wrapper(self, candidate, reference):
        return self.tracker.meteor(candidate, reference)

    def _hallucination_wrapper(self, candidate, reference):
        import nltk
        ref_tokens = nltk.word_tokenize(reference)
        return self.tracker.hallucination_rate(candidate, ref_tokens)

    def _f1_wrapper(self, candidate, reference):
        import nltk
        ref_tokens = nltk.word_tokenize(reference)
        cand_tokens = nltk.word_tokenize(candidate)
        return self.tracker.f1_precision_recall(ref_tokens, cand_tokens)

    def _coverage_wrapper(self, candidate, reference):
        import nltk
        ref_tokens = nltk.word_tokenize(reference)
        return self.tracker.coverage_score(candidate, ref_tokens)

    def compute(self, metrics, candidate, reference):
        results = {}

        for m in metrics:
            metric_fn = self.registry.get(m.lower())

            if not metric_fn:
                results[m] = {"error": "Unknown metric"}
                continue

            results[m] = metric_fn(candidate, reference)

        return results