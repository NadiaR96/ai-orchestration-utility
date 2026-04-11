class Scorer:
    def score(self, result: dict, strategy: str = "balanced") -> float:
        eval_metrics = result.get("evaluation", {})

        # fallback safety
        bert = eval_metrics.get("BERTScore", 0)
        hallucination = eval_metrics.get("HallucinationRate", 1)
        cost = result.get("cost", 1)
        latency = result.get("latency", 1)

        # -------------------------
        # Strategies
        # -------------------------
        quality_score = bert-hallucination

        if strategy == "quality":
            return quality_score

        elif strategy == "cost_aware":
            return quality_score * (1 / (1 + cost))

        elif strategy == "latency_aware":
            return quality_score * (1 / (1 + latency))

        elif strategy == "balanced":
            return quality_score * (1 / (1 + 0.05*cost + 0.05*latency))

        return 0.0