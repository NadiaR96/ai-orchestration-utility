from backend.core.types import EvaluationResult


class Comparator:
    def compare(self, a: EvaluationResult, b: EvaluationResult):

        if a.score > b.score:
            winner = "A"
        elif b.score > a.score:
            winner = "B"
        else:
            winner = "tie"

        return {
            "winner": winner,
            "score_breakdown": {
                "A": a.score,
                "B": b.score
            },
            "strategy": a.strategy
        }