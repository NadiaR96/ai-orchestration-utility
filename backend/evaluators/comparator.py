from backend.core.types import ComparisonResult


class Comparator:
    def compare_many(self, evaluations: dict, strategy: str = "balanced") -> ComparisonResult:

        scored = [
            (model, eval_result.score)
            for model, eval_result in evaluations.items()
        ]

        scored.sort(key=lambda x: x[1], reverse=True)

        ranking = [m for m, _ in scored]

        return ComparisonResult(
            winner=ranking[0],
            ranking=ranking,
            strategy=strategy,
            score_breakdown={
                model: eval_result.score
                for model, eval_result in evaluations.items()
            }
        )