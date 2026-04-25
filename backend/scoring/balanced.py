from backend.scoring.base import BaseScorer, output_length_penalty


class BalancedScorer(BaseScorer):
    def compute(self, metrics: dict) -> float:
        score = (
            (0.5 * metrics.get("bert_score", 0.0))
            + (0.3 * metrics.get("faithfulness", 0.0))
            - (0.3 * metrics.get("hallucination", 0.0))
            - (0.05 * metrics.get("cost_norm", 0.0))
            - (0.15 * metrics.get("latency_norm", 0.0))
        )

        return score - output_length_penalty(metrics)