from backend.core.types import ComparisonResult


class Comparator:
    def compare_many(self, evaluations: dict, strategy: str = "balanced") -> ComparisonResult:

        scored = [
            (model, eval_result.score)
            for model, eval_result in evaluations.items()
        ]

        scored.sort(key=lambda x: x[1], reverse=True)

        ranking = [m for m, _ in scored]
        top_score = scored[0][1]
        epsilon = 1e-9
        tied_winners = [model for model, score in scored if abs(score - top_score) <= epsilon]
        winner = tied_winners[0] if len(tied_winners) == 1 else None

        return ComparisonResult(
            winner=winner,
            ranking=ranking,
            strategy=strategy,
            score_breakdown={
                model: eval_result.score
                for model, eval_result in evaluations.items()
            },
            tied_winners=tied_winners if len(tied_winners) > 1 else []
        )