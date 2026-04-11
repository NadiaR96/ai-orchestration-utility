from backend.evaluators.scorer import Scorer


def compare_results(result_a: dict, result_b: dict, strategy: str = "balanced"):
    scorer = Scorer()

    score_a = scorer.score(result_a, strategy)
    score_b = scorer.score(result_b, strategy)

    if score_a > score_b:
        winner = "A"
    elif score_b > score_a:
        winner = "B"
    else:
        winner = "tie"

    return {
        "winner": winner,
        "strategy": strategy,
        "score_breakdown": {
            "A": round(score_a, 4),
            "B": round(score_b, 4)
        },
        "reason": _build_reason(result_a, result_b, strategy)
    }


def _build_reason(a, b, strategy):
    if strategy == "cost_aware":
        return "Winner chosen based on quality-to-cost ratio"

    if strategy == "latency_aware":
        return "Winner chosen based on quality-to-latency ratio"

    if strategy == "quality":
        return "Winner chosen based on output quality"

    return "Balanced scoring across quality, cost, and latency"