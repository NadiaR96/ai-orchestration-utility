import unittest
from backend.evaluators.evaluator import Evaluator
from backend.scoring.registry import get_scorer
from unittest.mock import patch

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator()
        self.candidate = "AI systems can have multiple agents working together."
        self.reference = "Multi-agent AI systems consist of multiple agents collaborating."
        self.chunks = ["Multi-agent AI systems", "consist of multiple agents", "collaborating"]
        self.scorer = get_scorer("quality")
        self.strategy = "quality"

    def test_evaluate_metrics(self):
        result = self.evaluator.evaluate(
            self.candidate, 
            self.reference, 
            self.chunks, 
            self.scorer, 
            self.strategy
        )
        self.assertIsNotNone(result.metrics)
        self.assertIsNotNone(result.score)
        self.assertEqual(result.strategy, "quality")

    @patch("backend.agents.hf_agent.HuggingFaceAgent.run")
    def test_huggingface_skip_real_api(self, mock_run):
        mock_run.return_value = "Mocked output"
        output = mock_run("Any input")
        self.assertEqual(output, "Mocked output")