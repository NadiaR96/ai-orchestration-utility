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
        prompt_token_count=None,
        output_token_count=None,
        total_token_count=None,
    ):
        raw = self.metrics.compute_all(output, reference, chunks)

        raw["cost"] = cost
        raw["latency"] = latency
        derived_output_tokens = float(len((output or "").split()))
        prompt_tokens = float(prompt_token_count if isinstance(prompt_token_count, (int, float)) else 0.0)
        output_tokens = float(output_token_count if isinstance(output_token_count, (int, float)) else derived_output_tokens)
        total_tokens = float(total_token_count if isinstance(total_token_count, (int, float)) else (prompt_tokens + output_tokens))

        raw["prompt_token_count"] = prompt_tokens
        raw["output_token_count"] = output_tokens
        raw["total_token_count"] = total_tokens

        quality_signal = (
            (0.5 * float(raw.get("bert_score", 0.0)))
            + (0.3 * float(raw.get("faithfulness", 0.0)))
        )
        raw["cost_per_1k_tokens"] = (float(cost) / max(total_tokens, 1.0)) * 1000.0
        raw["quality_per_1k_tokens"] = quality_signal / max(total_tokens / 1000.0, 0.001)

        norm = self.normaliser.normalise(raw)

        if strategy is None:
            strategy_name = scorer_or_strategy if isinstance(scorer_or_strategy, str) else "balanced"
            scorer = get_scorer(scorer_or_strategy)
        else:
            strategy_name = strategy
            scorer = get_scorer(strategy)

        score = max(0.0, min(1.0, scorer.compute(norm)))

        return EvaluationResult(
            metrics=norm,
            score=score,
            strategy=strategy_name
        )