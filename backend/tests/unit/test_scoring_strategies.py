import unittest
from backend.scoring.quality import QualityScorer
from backend.scoring.cost_aware import CostAwareScorer
from backend.scoring.balanced import BalancedScorer
from backend.scoring.rag_aware import RAGScorer
from backend.scoring.latency_aware import LatencyAwareScorer


class TestScoringStrategies(unittest.TestCase):
    def setUp(self):
        self.quality_scorer = QualityScorer()
        self.cost_aware_scorer = CostAwareScorer()
        self.balanced_scorer = BalancedScorer()
        self.rag_scorer = RAGScorer()
        self.latency_aware_scorer = LatencyAwareScorer()

    def test_quality_scorer_basic(self):
        metrics = {
            "bert_score": 0.8,
            "faithfulness": 0.9,
            "hallucination": 0.1
        }
        score = self.quality_scorer.compute(metrics)
        expected = (0.5 * 0.8) + (0.3 * 0.9) - (0.5 * 0.1)
        self.assertAlmostEqual(score, expected, places=5)

    def test_quality_scorer_missing_keys(self):
        metrics = {"bert_score": 0.7}
        score = self.quality_scorer.compute(metrics)
        expected = (0.5 * 0.7) + (0.3 * 0.0) - (0.5 * 0.0)
        self.assertAlmostEqual(score, expected, places=5)

    def test_cost_aware_scorer_basic(self):
        metrics = {
            "bert_score": 0.8,
            "faithfulness": 0.9,
            "hallucination": 0.1,
            "cost_norm": 0.2,
            "latency_norm": 0.3
        }
        score = self.cost_aware_scorer.compute(metrics)
        base = (0.6 * 0.8) + (0.4 * 0.9) - (0.3 * 0.1)
        expected = base - (0.2 * 0.2) - (0.2 * 0.3)
        self.assertAlmostEqual(score, expected, places=5)

    def test_balanced_scorer_basic(self):
        metrics = {
            "bert_score": 0.8,
            "faithfulness": 0.9,
            "hallucination": 0.1,
            "cost_norm": 0.2,
            "latency_norm": 0.3
        }
        score = self.balanced_scorer.compute(metrics)
        expected = (0.5 * 0.8) + (0.3 * 0.9) - (0.3 * 0.1) - (0.05 * 0.2) - (0.15 * 0.3)
        self.assertAlmostEqual(score, expected, places=5)

    def test_rag_scorer_basic(self):
        metrics = {
            "bert_score": 0.8,
            "faithfulness": 0.9,
            "hallucination": 0.1,
            "context_used": 0.7
        }
        score = self.rag_scorer.compute(metrics)
        expected = (0.4 * 0.8) + (0.4 * 0.9) + (0.2 * 0.7) - (0.3 * 0.1)
        self.assertAlmostEqual(score, expected, places=5)

    def test_latency_aware_scorer_basic(self):
        metrics = {
            "bert_score": 0.8,
            "faithfulness": 0.9,
            "hallucination": 0.1,
            "cost_norm": 0.2,
            "latency_norm": 0.3,
        }
        score = self.latency_aware_scorer.compute(metrics)
        expected = (0.35 * 0.8) + (0.25 * 0.9) - (0.30 * 0.1) - (0.55 * 0.3) - (0.05 * 0.2)
        self.assertAlmostEqual(score, expected, places=5)

    def test_latency_aware_prioritizes_faster_model(self):
        slower = {
            "bert_score": 0.85,
            "faithfulness": 0.9,
            "hallucination": 0.1,
            "cost_norm": 0.2,
            "latency_norm": 0.8,
        }
        faster = {
            "bert_score": 0.8,
            "faithfulness": 0.88,
            "hallucination": 0.1,
            "cost_norm": 0.2,
            "latency_norm": 0.2,
        }

        self.assertGreater(
            self.latency_aware_scorer.compute(faster),
            self.latency_aware_scorer.compute(slower),
        )

    def test_all_scorers_with_zero_metrics(self):
        metrics = {}
        self.assertEqual(self.quality_scorer.compute(metrics), 0.0)
        self.assertEqual(self.cost_aware_scorer.compute(metrics), 0.0)
        self.assertEqual(self.balanced_scorer.compute(metrics), 0.0)
        self.assertEqual(self.rag_scorer.compute(metrics), 0.0)
        self.assertEqual(self.latency_aware_scorer.compute(metrics), 0.0)

    def test_all_scorers_with_perfect_metrics(self):
        metrics = {
            "bert_score": 1.0,
            "faithfulness": 1.0,
            "hallucination": 0.0,
            "cost_norm": 0.0,
            "latency_norm": 0.0,
            "context_used": 1.0
        }
        self.assertAlmostEqual(self.quality_scorer.compute(metrics), 0.8, places=5)  # 0.5 + 0.3
        self.assertAlmostEqual(self.cost_aware_scorer.compute(metrics), 1.0, places=5)  # 1.0 + 0
        self.assertAlmostEqual(self.balanced_scorer.compute(metrics), 0.8, places=5)  # 0.5 + 0.3
        self.assertAlmostEqual(self.rag_scorer.compute(metrics), 1.0, places=5)  # 0.4 + 0.4 + 0.2
        self.assertAlmostEqual(self.latency_aware_scorer.compute(metrics), 0.6, places=5)  # 0.35 + 0.25

    def test_scorers_with_negative_values(self):
        metrics = {
            "bert_score": -0.1,
            "faithfulness": -0.2,
            "hallucination": 0.5,
            "cost_norm": 1.0,
            "latency_norm": 1.0,
            "context_used": -0.1
        }
        # Should handle negative values gracefully
        score = self.quality_scorer.compute(metrics)
        self.assertIsInstance(score, float)
        score = self.cost_aware_scorer.compute(metrics)
        self.assertIsInstance(score, float)
        score = self.balanced_scorer.compute(metrics)
        self.assertIsInstance(score, float)
        score = self.rag_scorer.compute(metrics)
        self.assertIsInstance(score, float)
        score = self.latency_aware_scorer.compute(metrics)
        self.assertIsInstance(score, float)

    def test_empty_output_penalized(self):
        metrics = {
            "bert_score": 0.9,
            "faithfulness": 0.9,
            "hallucination": 0.0,
            "cost_norm": 0.0,
            "latency_norm": 0.0,
            "context_used": 1.0,
            "output_token_count": 0,
        }

        # Baselines without penalty would be: quality/balanced=0.72, cost_aware/rag=0.9/0.96
        self.assertAlmostEqual(self.quality_scorer.compute(metrics), 0.32, places=5)
        self.assertAlmostEqual(self.balanced_scorer.compute(metrics), 0.32, places=5)
        self.assertAlmostEqual(self.cost_aware_scorer.compute(metrics), 0.5, places=5)
        self.assertAlmostEqual(self.rag_scorer.compute(metrics), 0.52, places=5)
        self.assertAlmostEqual(self.latency_aware_scorer.compute(metrics), 0.14, places=5)

    def test_short_output_penalized_less_than_empty(self):
        empty = {
            "bert_score": 0.8,
            "faithfulness": 0.8,
            "hallucination": 0.0,
            "output_token_count": 0,
        }
        short = {
            "bert_score": 0.8,
            "faithfulness": 0.8,
            "hallucination": 0.0,
            "output_token_count": 3,
        }

        self.assertLess(self.quality_scorer.compute(empty), self.quality_scorer.compute(short))
        self.assertLess(self.balanced_scorer.compute(empty), self.balanced_scorer.compute(short))


if __name__ == '__main__':
    unittest.main()