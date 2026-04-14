import unittest
from backend.evaluators.normaliser import Normaliser


class TestNormaliser(unittest.TestCase):
    def setUp(self):
        self.normaliser = Normaliser()

    def test_normalise_basic_metrics(self):
        metrics = {
            "bert_score": 0.8,
            "faithfulness": 0.9,
            "hallucination": 0.1,
            "cost": 0.05,
            "latency": 2.0
        }
        result = self.normaliser.normalise(metrics)

        # Check all original keys preserved
        for key in metrics:
            self.assertIn(key, result)
            self.assertEqual(result[key], metrics[key])

        # Check normalized values
        self.assertEqual(result["cost_norm"], 0.05)  # min(0.05, 1.0)
        self.assertEqual(result["latency_norm"], 2.0 / 5.0)  # min(2.0/5.0, 1.0)
        expected_quality = 0.5 * 0.8 + 0.3 * 0.9
        self.assertAlmostEqual(result["quality_norm"], expected_quality, places=5)

    def test_normalise_rouge_dict(self):
        metrics = {
            "rouge": {"rouge1": 0.8, "rougeL": 0.6},
            "bert_score": 0.7
        }
        result = self.normaliser.normalise(metrics)

        # ROUGE should be averaged
        self.assertEqual(result["rouge"], (0.8 + 0.6) / 2)
        self.assertEqual(result["bert_score"], 0.7)

    def test_normalise_tensor_like(self):
        # Mock tensor-like object
        class MockTensor:
            def item(self):
                return 0.85

        metrics = {
            "bert_score": MockTensor(),
            "cost": 0.1
        }
        result = self.normaliser.normalise(metrics)

        self.assertEqual(result["bert_score"], 0.85)
        self.assertEqual(result["cost"], 0.1)

    def test_normalise_invalid_types(self):
        metrics = {
            "bert_score": "invalid",
            "faithfulness": [1, 2, 3],
            "cost": 0.2
        }
        result = self.normaliser.normalise(metrics)

        # Invalid types should become 0.0
        self.assertEqual(result["bert_score"], 0.0)
        self.assertEqual(result["faithfulness"], 0.0)
        self.assertEqual(result["cost"], 0.2)

    def test_normalise_missing_keys(self):
        metrics = {"bert_score": 0.8}
        result = self.normaliser.normalise(metrics)

        # Missing keys should not be added, only existing ones processed
        self.assertNotIn("cost", result)
        self.assertNotIn("latency", result)
        self.assertIn("bert_score", result)

        # Normalized values should use defaults (0.0)
        self.assertEqual(result["cost_norm"], 0.0)
        self.assertEqual(result["latency_norm"], 0.0)
        expected_quality = 0.5 * 0.8 + 0.3 * 0.0
        self.assertAlmostEqual(result["quality_norm"], expected_quality, places=5)

    def test_normalise_high_cost_latency(self):
        metrics = {
            "cost": 2.0,  # Above 1.0
            "latency": 10.0,  # Above 5.0
            "bert_score": 0.9,
            "faithfulness": 0.8
        }
        result = self.normaliser.normalise(metrics)

        self.assertEqual(result["cost_norm"], 1.0)  # min(2.0, 1.0)
        self.assertEqual(result["latency_norm"], 1.0)  # min(10.0/5.0, 1.0) = min(2.0, 1.0)

    def test_normalise_empty_dict(self):
        metrics = {}
        result = self.normaliser.normalise(metrics)

        # All should be 0.0
        self.assertEqual(result["cost_norm"], 0.0)
        self.assertEqual(result["latency_norm"], 0.0)
        self.assertEqual(result["quality_norm"], 0.0)

    def test_normalise_quality_norm_calculation(self):
        metrics = {
            "bert_score": 1.0,
            "faithfulness": 1.0,
            "hallucination": 0.0
        }
        result = self.normaliser.normalise(metrics)

        # quality_norm = 0.5 * 1.0 + 0.3 * 1.0 = 0.8
        self.assertAlmostEqual(result["quality_norm"], 0.8, places=5)


if __name__ == '__main__':
    unittest.main()