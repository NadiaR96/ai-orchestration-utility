from backend.scoring.base import BaseScorer, output_length_penalty


class CostAwareScorer(BaseScorer):
    def compute(self, metrics: dict) -> float:
        bert = metrics.get("bert_score", 0.0)
        faith = metrics.get("faithfulness", 0.0)
        halluc = metrics.get("hallucination", 0.0)

        cost = metrics.get("cost_norm", 0.0)
        latency = metrics.get("latency_norm", 0.0)

        base = (
            (0.6 * bert)
            + (0.4 * faith)
            - (0.3 * halluc)
        )

        score = (
            base
            - (0.2 * cost)
            - (0.2 * latency)
        )

        return score - output_length_penalty(metrics)