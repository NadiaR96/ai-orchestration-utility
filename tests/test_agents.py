import unittest
from unittest.mock import patch
from agents.hf_agent import HuggingFaceAgent
from agents.evaluator_agent import EvaluatorAgent

class TestHuggingFaceAgent(unittest.TestCase):
    def setUp(self):
        self.agent = HuggingFaceAgent()

    @patch.object(HuggingFaceAgent, "run", return_value="This is a mocked response")
    def test_run_returns_string(self, mock_run):
        output = self.agent.run("Test input")
        self.assertEqual(output, "This is a mocked response")
        mock_run.assert_called_once_with("Test input")


class TestEvaluatorAgent(unittest.TestCase):
    def setUp(self):
        self.eval = EvaluatorAgent()

    def test_evaluate_returns_dict(self):
        result = self.eval.evaluate("input text", "output text")
        self.assertIsInstance(result, dict)
        self.assertIn("score", result)