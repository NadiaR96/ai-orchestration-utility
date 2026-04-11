from backend.metrics.metrics_tracker import MetricsTracker
from backend.evaluators.metrics_registry import MetricRegistry


class Evaluator:
    def __init__(self):
        self.tracker = MetricsTracker()
        self.registry = MetricRegistry(self.tracker)

    def evaluate(self, candidate, reference=None, metrics=None):
        if not reference:
            return {"note": "No reference provided - evaluation skipped."}

        if not metrics:
            metrics = ["meteor", "rouge", "bert_score"]

        return self.registry.compute(metrics, candidate, reference)


