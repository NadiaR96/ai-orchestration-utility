from backend.scoring.base import BaseScorer, output_length_penalty


class LatencyAwareScorer(BaseScorer):
    def compute(self, metrics: dict) -> float:
        bert = metrics.get("bert_score", 0.0)
        faith = metrics.get("faithfulness", 0.0)
        halluc = metrics.get("hallucination", 0.0)

        cost = metrics.get("cost_norm", 0.0)
        latency = metrics.get("latency_norm", 0.0)

        score = (
            (0.35 * bert)
            + (0.25 * faith)
            - (0.30 * halluc)
            - (0.55 * latency)
            - (0.05 * cost)
        )

        return score - output_length_penalty(metrics)