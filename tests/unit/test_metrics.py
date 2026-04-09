import unittest
from metrics.metrics_tracker import MetricsTracker

class TestMetricsTracker(unittest.TestCase):
    def setUp(self):
        self.metrics = MetricsTracker()
        self.candidate = "AI systems can have multiple agents working together."
        self.reference = "Multi-agent AI systems consist of multiple agents collaborating."

    def test_meteor(self):
        score = self.metrics.meteor(self.candidate, self.reference)
        self.assertIsInstance(score, float)

    def test_rouge(self):
        score = self.metrics.rouge(self.candidate, self.reference)
        self.assertIn('rouge1', score)
        self.assertIn('rougeL', score)

    def test_bert_score(self):
        score = self.metrics.bert_score([self.candidate], [self.reference])
        self.assertIsInstance(score, float)

    def test_perplexity(self):
        score = self.metrics.perplexity(self.candidate, self.reference)
        self.assertIsInstance(score, float)

    def test_hallucination_rate(self):
        ref_tokens = self.reference.split()
        rate = self.metrics.hallucination_rate(self.candidate, ref_tokens)
        self.assertIsInstance(rate, float)

    def test_diversity_score(self):
        score = self.metrics.diversity_score(self.candidate)
        self.assertIsInstance(score, float)

    def test_latency(self):
        import time
        start, end = time.time(), time.time() + 1
        latency = self.metrics.latency(start, end)
        self.assertEqual(latency, 1)

    def test_cost_estimate(self):
        cost = self.metrics.cost_estimate(1000)
        self.assertEqual(cost, 0.01)