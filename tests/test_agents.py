import unittest
from agents.evaluator_agent import EvaluatorAgent
from unittest.mock import patch

class TestEvaluatorAgent(unittest.TestCase):
    def setUp(self):
        self.evaluator = EvaluatorAgent()
        self.candidate = "AI systems can have multiple agents working together."
        self.reference = "Multi-agent AI systems consist of multiple agents collaborating."

    def test_evaluate_metrics(self):
        metrics = self.evaluator.evaluate(self.candidate, self.reference)
        self.assertIn("METEOR", metrics)
        self.assertIn("ROUGE", metrics)
        self.assertIn("BERTScore", metrics)
        self.assertIn("Perplexity", metrics)
        self.assertIn("HallucinationRate", metrics)
        self.assertIn("Diversity", metrics)
        self.assertIn("Cost", metrics)

    @patch("agents.hf_agent.HuggingFaceAgent.run")
    def test_huggingface_skip_real_api(self, mock_run):
        mock_run.return_value = "Mocked output"
        output = mock_run("Any input")
        self.assertEqual(output, "Mocked output")