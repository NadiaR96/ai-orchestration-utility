import unittest
from backend.orchestrator import Orchestrator
from unittest.mock import patch, MagicMock


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        # Mock model output
        patcher = patch(
            "backend.agents.hf_agent.HuggingFaceAgent.run",
            return_value="Mocked output"
        )
        self.addCleanup(patcher.stop)
        self.mock_run = patcher.start()

        # Mock retriever
        retriever_patcher = patch("backend.orchestrator.Retriever")
        mock_retriever_class = retriever_patcher.start()
        self.addCleanup(retriever_patcher.stop)

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_retriever_class.return_value = mock_retriever

        self.orch = Orchestrator()

    def test_process_task_without_reference(self):
        result = self.orch.process_task("What is RAG?")

        self.assertIsNotNone(result.output)
        self.assertIsNotNone(result.evaluation)

        # Should still produce metrics
        self.assertIn("faithfulness", result.evaluation.metrics)

        # Reference-based metrics should safely exist (but be 0 or default)
        self.assertIn("bert_score", result.evaluation.metrics)
        
    def test_process_task_with_reference(self):
        result = self.orch.process_task({
            "input": "What is RAG?",
            "reference": "RAG stands for Retrieval Augmented Generation"
        })

        self.assertIsNotNone(result.output)
        self.assertIsNotNone(result.evaluation)

        # Now metrics should be meaningful
        self.assertGreaterEqual(result.evaluation.metrics["bert_score"], 0.0)
        self.assertIn("rouge", result.evaluation.metrics)