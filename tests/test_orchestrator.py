import unittest
from orchestrator import Orchestrator
from unittest.mock import patch

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        # Patch HuggingFaceAgent to avoid real API calls
        patcher = patch("agents.hf_agent.HuggingFaceAgent.run", return_value="Mocked output")
        self.addCleanup(patcher.stop)
        self.mock_run = patcher.start()
        self.orch = Orchestrator()

    def test_process_task_returns_metrics(self):
        task = {"prompt": "Explain multi-agent AI systems"}
        result = self.orch.process_task(task)
        self.assertIn("METEOR", result)
        self.assertIn("ROUGE", result)
        self.assertIn("BERTScore", result)
        self.assertIn("Perplexity", result)
        self.assertIn("HallucinationRate", result)
        self.assertIn("Diversity", result)
        self.assertIn("Cost", result)