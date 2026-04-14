from backend.core.types import EvaluationResult
from backend.metrics.metrics_tracker import MetricsTracker
from backend.evaluators.normaliser import Normaliser


class Evaluator:
    def __init__(self):
        self.metrics = MetricsTracker()
        self.normaliser = Normaliser()

    def evaluate(self, output, reference, chunks, scorer, strategy, cost=0.0, latency=0.0):

        raw = self.metrics.compute_all(output, reference, chunks)

        raw["cost"] = cost
        raw["latency"] = latency

        norm = self.normaliser.normalise(raw)

        score = scorer.compute(norm)

        return EvaluationResult(
            metrics=norm,
            score=score,
            strategy=strategy
        )