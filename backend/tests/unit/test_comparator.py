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

        result = self.comparator.compare(a, b)

        self.assertEqual(result["winner"], "A")
        self.assertEqual(result["score_breakdown"]["A"], 0.85)
        self.assertEqual(result["score_breakdown"]["B"], 0.75)
        self.assertEqual(result["strategy"], "quality")

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

        result = self.comparator.compare(a, b)

        self.assertEqual(result["winner"], "B")
        self.assertEqual(result["score_breakdown"]["A"], 0.65)
        self.assertEqual(result["score_breakdown"]["B"], 0.85)

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

        result = self.comparator.compare(a, b)

        self.assertEqual(result["winner"], "tie")
        self.assertEqual(result["score_breakdown"]["A"], 0.8)
        self.assertEqual(result["score_breakdown"]["B"], 0.8)

    def test_compare_different_strategies(self):
        # Strategy should come from A
        a = EvaluationResult(
            metrics={"bert_score": 0.9},
            score=0.9,
            strategy="rag_aware"
        )
        b = EvaluationResult(
            metrics={"bert_score": 0.7},
            score=0.7,
            strategy="quality"  # Different, but should use A's
        )

        result = self.comparator.compare(a, b)

        self.assertEqual(result["winner"], "A")
        self.assertEqual(result["strategy"], "rag_aware")

    def test_compare_zero_scores(self):
        a = EvaluationResult(
            metrics={},
            score=0.0,
            strategy="quality"
        )
        b = EvaluationResult(
            metrics={},
            score=0.0,
            strategy="quality"
        )

        result = self.comparator.compare(a, b)

        self.assertEqual(result["winner"], "tie")
        self.assertEqual(result["score_breakdown"]["A"], 0.0)
        self.assertEqual(result["score_breakdown"]["B"], 0.0)

    def test_compare_negative_scores(self):
        a = EvaluationResult(
            metrics={"bert_score": -0.1},
            score=-0.1,
            strategy="quality"
        )
        b = EvaluationResult(
            metrics={"bert_score": -0.2},
            score=-0.2,
            strategy="quality"
        )

        result = self.comparator.compare(a, b)

        self.assertEqual(result["winner"], "A")  # -0.1 > -0.2
        self.assertEqual(result["score_breakdown"]["A"], -0.1)
        self.assertEqual(result["score_breakdown"]["B"], -0.2)


if __name__ == '__main__':
    unittest.main()