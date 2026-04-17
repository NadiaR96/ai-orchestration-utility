import unittest
from backend.api import leaderboard
from backend.core.types import EvaluationResult, RunResult


class TestLeaderboardHelpers(unittest.TestCase):
    def _make_run(self, model, latency, cost):
        return RunResult(
            output=f"Output {model}",
            model=model,
            retrieval="rag",
            latency=latency,
            cost=cost,
            context_used=True,
            rag_context={},
        )

    def _make_eval(self, score):
        return EvaluationResult(
            metrics={
                "bert_score": score,
                "faithfulness": score,
                "hallucination": max(0.0, 1.0 - score),
                "cost_norm": 0.1,
                "latency_norm": 0.1,
                "context_used": 1.0,
            },
            score=score,
            strategy="balanced",
        )

    def test_rank_and_paginate_all_strategies(self):
        runs = {
            "small": self._make_run("small", latency=1.0, cost=0.01),
            "large": self._make_run("large", latency=1.5, cost=0.02),
        }
        evaluations = {
            "small": self._make_eval(0.8),
            "large": self._make_eval(0.9),
        }

        entries = leaderboard._build_entries_from_runs(runs, evaluations)
        response = leaderboard._rank_and_paginate(entries, "balanced", page=1, page_size=10)

        self.assertEqual(response.total_items, 2)
        self.assertFalse(response.has_more)
        self.assertEqual(response.items[0].model, "large")
        self.assertIn("balanced", response.strategy_rankings)
        self.assertIn("quality", response.strategy_rankings)
        self.assertIn("cost_aware", response.strategy_rankings)
        self.assertIn("rag", response.strategy_rankings)

        for item in response.items:
            self.assertIn("balanced", item.ranks_by_strategy)
            self.assertIn("quality", item.ranks_by_strategy)
            self.assertIn("cost_aware", item.ranks_by_strategy)
            self.assertIn("rag", item.ranks_by_strategy)
            self.assertTrue(item.narrative)

    def test_rank_and_paginate_page_metadata(self):
        runs = {
            "a": self._make_run("a", latency=1.0, cost=0.01),
            "b": self._make_run("b", latency=1.2, cost=0.02),
            "c": self._make_run("c", latency=1.3, cost=0.03),
        }
        evaluations = {
            "a": self._make_eval(0.95),
            "b": self._make_eval(0.85),
            "c": self._make_eval(0.75),
        }

        entries = leaderboard._build_entries_from_runs(runs, evaluations)
        response = leaderboard._rank_and_paginate(entries, "balanced", page=2, page_size=1)

        self.assertEqual(response.page, 2)
        self.assertEqual(response.page_size, 1)
        self.assertEqual(response.total_items, 3)
        self.assertTrue(response.has_more)
        self.assertEqual(response.next_page, 3)
        self.assertEqual(len(response.items), 1)

    def test_rank_tie_breaker_latency_then_cost_then_name(self):
        runs = {
            "beta": self._make_run("beta", latency=1.1, cost=0.01),
            "alpha": self._make_run("alpha", latency=1.1, cost=0.01),
            "gamma": self._make_run("gamma", latency=0.9, cost=0.02),
        }

        tie_eval = self._make_eval(0.8)
        evaluations = {
            "beta": tie_eval,
            "alpha": tie_eval,
            "gamma": tie_eval,
        }

        entries = leaderboard._build_entries_from_runs(runs, evaluations)
        response = leaderboard._rank_and_paginate(entries, "balanced", page=1, page_size=10)

        # All scores tie, so latency asc first, then cost asc, then model name.
        ordered = [item.model for item in response.items]
        self.assertEqual(ordered, ["gamma", "alpha", "beta"])


if __name__ == "__main__":
    unittest.main()
