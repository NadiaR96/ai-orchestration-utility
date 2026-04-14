import unittest
from unittest.mock import patch
from backend.metrics.metrics_tracker import MetricsTracker

class TestMetricsTracker(unittest.TestCase):
    def setUp(self):
        self.metrics = MetricsTracker()
        self.candidate = "AI systems can have multiple agents working together."
        self.reference = "Multi-agent AI systems consist of multiple agents collaborating."
        self.chunks = ["Multi-agent AI systems", "consist of multiple agents", "collaborating"]

    def test_rouge(self):
        score = self.metrics.rouge(self.candidate, self.reference)
        self.assertIn('rouge1', score)
        self.assertIn('rougeL', score)
        self.assertIsInstance(score['rouge1'], float)
        self.assertIsInstance(score['rougeL'], float)
        self.assertGreaterEqual(score['rouge1'], 0.0)
        self.assertLessEqual(score['rouge1'], 1.0)

    @patch('backend.metrics.metrics_tracker.bert_score_fn')
    def test_bert_score(self, mock_bert):
        # Mock the BERT score function to return (P, R, F1)
        from unittest.mock import MagicMock
        mock_f1 = MagicMock()
        mock_f1.mean.return_value.item.return_value = 0.85
        mock_bert.return_value = (None, None, mock_f1)

        score = self.metrics.bert_score(self.candidate, self.reference)
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 0.85, places=5)

    @patch('backend.metrics.metrics_tracker.bert_score_fn')
    def test_bert_score_exception(self, mock_bert):
        mock_bert.side_effect = Exception("BERT failed")
        score = self.metrics.bert_score(self.candidate, self.reference)
        self.assertEqual(score, 0.0)

    def test_bleu(self):
        # Test BLEU calculation
        candidate = "the cat sat"
        reference = "the cat sat on the mat"
        bleu = self.metrics.bleu(candidate, reference)
        # 3 common words out of 3 = 1.0
        self.assertEqual(bleu, 1.0)

    def test_bleu_no_overlap(self):
        candidate = "dog runs fast"
        reference = "cat sits slow"
        bleu = self.metrics.bleu(candidate, reference)
        self.assertEqual(bleu, 0.0)

    def test_hallucination_rate(self):
        ref_tokens = self.reference.split()
        rate = self.metrics.hallucination_rate(self.candidate, ref_tokens)
        self.assertIsInstance(rate, float)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_faithfulness(self):
        faith = self.metrics.faithfulness(self.candidate, self.chunks)
        self.assertIsInstance(faith, float)
        self.assertGreaterEqual(faith, 0.0)
        self.assertLessEqual(faith, 1.0)

    def test_diversity_score(self):
        text = "the cat cat sat"
        diversity = self.metrics.diversity_score(text)
        # Unique: the, cat, sat (3/4)
        self.assertAlmostEqual(diversity, 3/4, places=3)

    def test_diversity_score_empty(self):
        diversity = self.metrics.diversity_score("")
        self.assertEqual(diversity, 0.0)

    def test_perplexity(self):
        perplexity = self.metrics.perplexity(self.candidate, self.reference)
        self.assertIsInstance(perplexity, float)
        self.assertGreaterEqual(perplexity, 1.0)  # Minimum perplexity

    def test_perplexity_no_reference(self):
        perplexity = self.metrics.perplexity(self.candidate, "")
        self.assertEqual(perplexity, 0.0)

    @patch('backend.metrics.metrics_tracker.bert_score_fn')
    def test_compute_all_with_reference(self, mock_bert):
        mock_f1 = type('MockTensor', (), {'mean': lambda: type('MockMean', (), {'item': lambda: 0.8})()})()
        mock_bert.return_value = (None, None, mock_f1)

        results = self.metrics.compute_all(self.candidate, self.reference, self.chunks)

        required_keys = ["bert_score", "bleu", "rouge", "perplexity", "hallucination", "faithfulness", "diversity"]
        for key in required_keys:
            self.assertIn(key, results)
            if key != "rouge":
                self.assertIsInstance(results[key], float)

        self.assertIsInstance(results["rouge"], dict)

    @patch('backend.metrics.metrics_tracker.bert_score_fn')
    def test_compute_all_without_reference(self, mock_bert):
        # BERT should not be called
        results = self.metrics.compute_all(self.candidate, "", self.chunks)

        # Should have default values
        self.assertEqual(results["bert_score"], 0.0)
        self.assertEqual(results["bleu"], 0.0)
        self.assertEqual(results["rouge"], 0.0)
        self.assertEqual(results["perplexity"], 0.0)
        self.assertEqual(results["hallucination"], 0.0)

        # But still compute faithfulness and diversity
        self.assertIsInstance(results["faithfulness"], float)
        self.assertIsInstance(results["diversity"], float)