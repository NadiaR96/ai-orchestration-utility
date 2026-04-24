from backend.scoring.base import BaseScorer, output_length_penalty


class RAGScorer(BaseScorer):
    def compute(self, metrics: dict) -> float:
        bert = metrics.get("bert_score", 0.0)
        faith = metrics.get("faithfulness", 0.0)
        halluc = metrics.get("hallucination", 0.0)

        context_used = metrics.get("context_used", 0.0)

        score = (
            (0.4 * bert)
            + (0.4 * faith)
            + (0.2 * context_used)
            - (0.3 * halluc)
        )

        return score - output_length_penalty(metrics)