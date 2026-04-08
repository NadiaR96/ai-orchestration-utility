import unittest
from metrics.metrics_tracker import MetricsTracker

class TestMetricsTracker(unittest.TestCase):
    def setUp(self):
        self.metrics = MetricsTracker()
        self.reference = "AI generates useful insights"
        self.candidate = "AI creates useful insights"

    def test_bert_score(self):
        score = self.metrics.bert_score(self.candidate, self.reference)
        self.assertGreater(score, 0.8)  # semantic similarity should be high

    def test_diversity_score(self):
        score = self.metrics.diversity_score(self.candidate)
        self.assertGreater(score, 0)  # should be > 0

    def test_hallucination_rate(self):
        ref_tokens = self.reference.split()
        rate = self.metrics.hallucination_rate(self.candidate, ref_tokens)
        self.assertLessEqual(rate, 0.5)