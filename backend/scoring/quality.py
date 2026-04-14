from backend.scoring.base import BaseScorer


class QualityScorer(BaseScorer):
    def compute(self, metrics: dict) -> float:
        bert = metrics.get("bert_score", 0.0)
        faith = metrics.get("faithfulness", 0.0)
        halluc = metrics.get("hallucination", 0.0)

        return (
            (0.5 * bert)
            + (0.3 * faith)
            - (0.5 * halluc)
        )