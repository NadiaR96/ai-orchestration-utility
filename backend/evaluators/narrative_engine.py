class EvaluationNarrativeEngine:
    def explain(self, a: dict, b: dict, winner: str) -> dict:
        return {
            "A": self._explain_single(a),
            "B": self._explain_single(b),
            "comparison": self._compare(a, b, winner)
        }

    def _explain_single(self, r: dict) -> str:
        parts = []

        # quality
        if r.get("evaluation", {}).get("bert_score") is not None:
            parts.append(f"semantic quality={r['evaluation']['bert_score']:.3f}")

        if "hallucination" in r.get("evaluation", {}):
            parts.append(f"hallucination={r['evaluation']['hallucination']:.3f}")

        # efficiency
        if "cost" in r:
            parts.append(f"cost={r['cost']:.6f}")

        if "latency" in r:
            parts.append(f"latency={r['latency']:.3f}s")

        return " | ".join(parts)

    def _compare(self, a: dict, b: dict, winner: str) -> dict:
        a_cost = a.get("cost", 0)
        b_cost = b.get("cost", 0)

        a_lat = a.get("latency", 0)
        b_lat = b.get("latency", 0)

        return {
            "winner": winner,
            "tradeoffs": {
                "cost": "A cheaper" if a_cost < b_cost else "B cheaper",
                "latency": "A faster" if a_lat < b_lat else "B faster"
            },
            "interpretation": self._winner_reason(a, b, winner)
        }

    def _winner_reason(self, a: dict, b: dict, winner: str) -> str:
        if winner == "tie":
            return "Both models show similar tradeoffs across quality and efficiency."

        return f"{winner} selected based on combined quality, cost, and latency tradeoff."

    def selection_reason(
        self,
        selected_variant: str,
        strategy: str,
        best_score: float,
        fastest_variant: str,
        cheapest_variant: str,
        tie_variants: list | None = None,
    ) -> str:
        tie_variants = tie_variants or []

        if tie_variants:
            tied = ", ".join(tie_variants)
            return (
                f"Models selected (tie): {tied} based on {strategy} strategy and trade-offs "
                f"(best_score={best_score:.4f}, fastest={fastest_variant}, cheapest={cheapest_variant})."
            )

        return (
            f"Model selected: {selected_variant} based on {strategy} strategy and trade-offs "
            f"(best_score={best_score:.4f}, fastest={fastest_variant}, cheapest={cheapest_variant})."
        )