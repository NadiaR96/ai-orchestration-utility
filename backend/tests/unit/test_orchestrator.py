import unittest
from backend.orchestrator.orchestrator import Orchestrator
from unittest.mock import patch, MagicMock


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        # Mock retriever with proper signature.
        retriever_patcher = patch("backend.orchestrator.orchestrator.Retriever")
        mock_retriever_class = retriever_patcher.start()
        self.addCleanup(retriever_patcher.stop)

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        # Ensure the mock returns the instance when called.
        mock_retriever_class.return_value = mock_retriever

        # Mock model loading to avoid transformer weight warnings.
        model_patcher = patch("backend.orchestrator.orchestrator.get_model")
        mock_get_model = model_patcher.start()
        self.addCleanup(model_patcher.stop)
        mock_get_model.return_value = "mock-model"

        # Mock HuggingFaceAgent initialization.
        agent_patcher = patch("backend.orchestrator.orchestrator.HuggingFaceAgent")
        mock_agent_class = agent_patcher.start()
        self.addCleanup(agent_patcher.stop)
        mock_agent = MagicMock()
        mock_agent.run.return_value = "Mocked output"
        mock_agent_class.return_value = mock_agent

        self.orch = Orchestrator()

    def test_process_task_without_reference(self):
        result = self.orch.process_task("What is RAG?")

        self.assertIsNotNone(result.output)
        self.assertIsNotNone(result.evaluation)

        # Should still produce metrics.
        self.assertIn("faithfulness", result.evaluation.metrics)

        # Reference-based metrics should safely exist (but be 0 or default).
        self.assertIn("bert_score", result.evaluation.metrics)

    def test_process_task_with_reference(self):
        result = self.orch.process_task({
            "input": "What is RAG?",
            "reference": "RAG stands for Retrieval Augmented Generation"
        })

        self.assertIsNotNone(result.output)
        self.assertIsNotNone(result.evaluation)

        # Now metrics should be meaningful.
        self.assertGreaterEqual(result.evaluation.metrics["bert_score"], 0.0)
        self.assertIn("rouge", result.evaluation.metrics)

if __name__ == '__main__':
    unittest.main()
