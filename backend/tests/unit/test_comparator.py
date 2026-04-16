import unittest
from backend.evaluators.comparator import Comparator
from backend.core.types import EvaluationResult


class TestComparator(unittest.TestCase):
    def setUp(self):
        self.comparator = Comparator()

    def test_compare_a_wins(self):
        a = EvaluationResult(
            metrics={"bert_score": 0.8},
            score=0.85,
            strategy="quality"
        )
        b = EvaluationResult(
            metrics={"bert_score": 0.7},
            score=0.75,
            strategy="quality"
        )

        evaluations = {
            "A": a,
            "B": b
        }

        result = self.comparator.compare_many(evaluations)

        self.assertEqual(result.winner, "A")
        self.assertEqual(result.score_breakdown["A"], 0.85)
        self.assertEqual(result.score_breakdown["B"], 0.75)
        self.assertEqual(result.strategy, "balanced")

    def test_compare_b_wins(self):
        a = EvaluationResult(
            metrics={"bert_score": 0.6},
            score=0.65,
            strategy="cost_aware"
        )
        b = EvaluationResult(
            metrics={"bert_score": 0.8},
            score=0.85,
            strategy="cost_aware"
        )

        evaluations = {
            "A": a,
            "B": b
        }

        result = self.comparator.compare_many(evaluations)

        self.assertEqual(result.winner, "B")

    def test_compare_tie(self):
        a = EvaluationResult(
            metrics={"bert_score": 0.8},
            score=0.8,
            strategy="balanced"
        )
        b = EvaluationResult(
            metrics={"bert_score": 0.8},
            score=0.8,
            strategy="balanced"
        )

        evaluations = {
            "A": a,
            "B": b
        }

        result = self.comparator.compare_many(evaluations)

        # tie → first wins (stable sort)
        self.assertEqual(result.winner, "A")

    def test_compare_zero_scores(self):
        a = EvaluationResult(metrics={}, score=0.0, strategy="quality")
        b = EvaluationResult(metrics={}, score=0.0, strategy="quality")

        evaluations = {
            "A": a,
            "B": b
        }

        result = self.comparator.compare_many(evaluations)

        self.assertEqual(result.winner, "A")
        self.assertEqual(result.score_breakdown["A"], 0.0)
        self.assertEqual(result.score_breakdown["B"], 0.0)
    def test_compare_negative_scores(self):
        a = EvaluationResult(metrics={"bert_score": -0.1}, score=-0.1, strategy="quality")
        b = EvaluationResult(metrics={"bert_score": -0.2}, score=-0.2, strategy="quality")

        evaluations = {
            "A": a,
            "B": b
        }

        result = self.comparator.compare_many(evaluations)

        self.assertEqual(result.winner, "A")


if __name__ == '__main__':
    unittest.main()