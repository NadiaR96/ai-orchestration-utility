from backend.core.types import EvaluationResult
from backend.metrics.metrics_tracker import MetricsTracker
from backend.evaluators.normaliser import Normaliser
from backend.scoring.registry import get_scorer

class Evaluator:
    def __init__(self):
        self.metrics = MetricsTracker()
        self.normaliser = Normaliser()

    def evaluate(
        self,
        output,
        reference,
        chunks,
        scorer_or_strategy="balanced",
        strategy=None,
        cost=0.0,
        latency=0.0,
    ):
        raw = self.metrics.compute_all(output, reference, chunks)

        raw["cost"] = cost
        raw["latency"] = latency

        norm = self.normaliser.normalise(raw)

        if strategy is None:
            strategy_name = scorer_or_strategy if isinstance(scorer_or_strategy, str) else "balanced"
            scorer = get_scorer(scorer_or_strategy)
        else:
            strategy_name = strategy
            scorer = get_scorer(scorer_or_strategy)

        score = scorer.compute(norm)

        return EvaluationResult(
            metrics=norm,
            score=score,
            strategy=strategy_name
        )