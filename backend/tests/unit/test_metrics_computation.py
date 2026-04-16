import unittest
from unittest.mock import patch, MagicMock
from backend.metrics.metrics_tracker import MetricsTracker


class TestMetricsComputation(unittest.TestCase):
    def setUp(self):
        self.tracker = MetricsTracker()

    @patch('backend.metrics.metrics_tracker.bert_score')
    def test_compute_all_with_reference(self, mock_bert):
        # Mock BERT score to return tensor-like object
        mock_f1 = MagicMock()
        mock_f1.mean.return_value.item.return_value = 0.85
        mock_bert.return_value = (None, None, mock_f1)

        output = "The cat sat on the mat."
        reference = "A cat is sitting on a mat."
        chunks = ["The cat is an animal.", "Cats like to sit on mats."]

        results = self.tracker.compute_all(output, reference, chunks)

        # Check all keys exist
        expected_keys = ["bert_score", "bleu", "rouge", "perplexity", "hallucination", "faithfulness", "diversity"]
        for key in expected_keys:
            self.assertIn(key, results)

        # Check types
        self.assertIsInstance(results["bert_score"], float)
        self.assertIsInstance(results["bleu"], float)
        self.assertIsInstance(results["rouge"], dict)
        self.assertIsInstance(results["perplexity"], float)
        self.assertIsInstance(results["hallucination"], float)
        self.assertIsInstance(results["faithfulness"], float)
        self.assertIsInstance(results["diversity"], float)

        # Check ROUGE structure
        self.assertIn("rouge1", results["rouge"])
        self.assertIn("rougeL", results["rouge"])

    @patch('backend.metrics.metrics_tracker.bert_score')
    def test_compute_all_without_reference(self, mock_bert):
        output = "The cat sat on the mat."
        reference = ""
        chunks = ["The cat is an animal."]

        results = self.tracker.compute_all(output, reference, chunks)

        # BERT should not be called when there is no reference
        mock_bert.assert_not_called()

        # Check default values
        self.assertEqual(results["bert_score"], 0.0)
        self.assertEqual(results["bleu"], 0.0)
        self.assertEqual(results["rouge"], 0.0)
        self.assertEqual(results["perplexity"], 0.0)
        self.assertEqual(results["hallucination"], 0.0)

        # Faithfulness and diversity should still be computed
        self.assertIsInstance(results["faithfulness"], float)
        self.assertIsInstance(results["diversity"], float)

    def test_compute_all_with_list_reference(self):
        output = "Hello world"
        reference = ["Hello", "world", "test"]
        chunks = ["Hello there"]

        results = self.tracker.compute_all(output, reference, chunks)

        # Should join list into string
        self.assertIsInstance(results["bert_score"], float)

    def test_bleu_basic(self):
        output = "the cat sat"
        reference = "the cat sat on the mat"
        bleu = self.tracker.bleu(output, reference)
        # 3 common words out of 3 = 1.0
        self.assertEqual(bleu, 1.0)

    def test_bleu_no_overlap(self):
        output = "dog runs fast"
        reference = "cat sits slow"
        bleu = self.tracker.bleu(output, reference)
        # No common words = 0.0
        self.assertEqual(bleu, 0.0)

    def test_bleu_empty_output(self):
        output = ""
        reference = "some text"
        bleu = self.tracker.bleu(output, reference)
        # Empty output = 0.0 (denom = 1)
        self.assertEqual(bleu, 0.0)

    def test_hallucination_rate_basic(self):
        output = "the cat sat on the red mat"
        reference_tokens = ["the", "cat", "sat", "on", "the", "mat"]
        halluc = self.tracker.hallucination_rate(output, reference_tokens)
        # "red" is hallucinated, 1/6 ≈ 0.1667
        self.assertAlmostEqual(halluc, 1/6, places=3)

    def test_hallucination_rate_no_hallucination(self):
        output = "the cat sat"
        reference_tokens = ["the", "cat", "sat", "on", "mat"]
        halluc = self.tracker.hallucination_rate(output, reference_tokens)
        self.assertEqual(halluc, 0.0)

    def test_faithfulness_basic(self):
        output = "the cat sat on the mat"
        chunks = ["the cat is black", "cats sit on mats"]
        faith = self.tracker.faithfulness(output, chunks)
        # Common words: the, cat, on (3/5)
        self.assertAlmostEqual(faith, 3/5, places=3)

    def test_faithfulness_empty_chunks(self):
        output = "hello world"
        chunks = []
        faith = self.tracker.faithfulness(output, chunks)
        self.assertEqual(faith, 0.0)

    def test_diversity_score_basic(self):
        text = "the cat cat sat"
        diversity = self.tracker.diversity_score(text)
        # Unique tokens: the, cat, sat (3/4)
        self.assertAlmostEqual(diversity, 3/4, places=3)

    def test_diversity_score_all_unique(self):
        text = "the cat sat on mat"
        diversity = self.tracker.diversity_score(text)
        self.assertEqual(diversity, 1.0)

    def test_diversity_score_empty(self):
        text = ""
        diversity = self.tracker.diversity_score(text)
        self.assertEqual(diversity, 0.0)

    def test_perplexity_basic(self):
        candidate = "the cat sat"
        reference = "the cat sat on the mat"
        perplexity = self.tracker.perplexity(candidate, reference)
        # All tokens in reference, prob=1.0, perplexity=1.0
        self.assertEqual(perplexity, 1.0)

    def test_perplexity_no_overlap(self):
        candidate = "dog runs fast"
        reference = "cat sits slow"
        perplexity = self.tracker.perplexity(candidate, reference)
        # No overlap, prob=0.003 (clamped), perplexity≈333.33, capped at 50.0
        self.assertEqual(perplexity, 50.0)

    def test_perplexity_empty_candidate(self):
        candidate = ""
        reference = "some text"
        perplexity = self.tracker.perplexity(candidate, reference)
        self.assertEqual(perplexity, 0.0)

    def test_perplexity_empty_reference(self):
        candidate = "some text"
        reference = ""
        perplexity = self.tracker.perplexity(candidate, reference)
        self.assertEqual(perplexity, 0.0)

    @patch('backend.metrics.metrics_tracker.bert_score_fn')
    def test_bert_score_exception_handling(self, mock_bert):
        mock_bert.side_effect = Exception("Test error")
        result = self.tracker.bert_score("test", "ref")
        self.assertEqual(result, 0.0)

    def test_rouge_basic(self):
        output = "The cat sat on the mat"
        reference = "A cat is sitting on a mat"
        rouge = self.tracker.rouge(output, reference)
        self.assertIn("rouge1", rouge)
        self.assertIn("rougeL", rouge)
        self.assertIsInstance(rouge["rouge1"], float)
        self.assertIsInstance(rouge["rougeL"], float)


if __name__ == '__main__':
    unittest.main()