import unittest
from orchestrator import Orchestrator

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orch = Orchestrator()

    def test_process_task_returns_metrics(self):
        task = {
            "input": "Explain multi-agent AI systems in simple terms",
            "reference": "AI systems can have multiple agents for reasoning"
        }
        result = self.orch.process_task(task)
        self.assertIn("metrics", result)
        self.assertIn("BERTScore", result["metrics"])
        self.assertIn("Diversity", result["metrics"])