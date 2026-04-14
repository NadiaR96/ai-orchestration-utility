import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.main import app  # Assuming main.py creates the FastAPI app


class TestAPIIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    @patch('backend.api.run_task.orchestrator.process_task')
    def test_run_task_string_input(self, mock_process):
        from backend.core.types import RunResult, EvaluationResult

        mock_evaluation = EvaluationResult(
            metrics={"bert_score": 0.8},
            score=0.85,
            strategy="quality"
        )
        mock_result = RunResult(
            output="Generated response",
            evaluation=mock_evaluation,
            model="small",
            retrieval="rag",
            latency=1.2,
            cost=0.01,
            context_used=True,
            rag_context={}
        )
        mock_process.return_value = mock_result

        response = self.client.post("/run-task", json={"input": "Test prompt"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["output"], "Generated response")
        self.assertEqual(data["evaluation"]["score"], 0.85)
        mock_process.assert_called_once_with({"input": "Test prompt"})

    @patch('backend.api.run_task.orchestrator.process_task')
    def test_run_task_dict_input(self, mock_process):
        from backend.core.types import RunResult, EvaluationResult

        mock_evaluation = EvaluationResult(
            metrics={"bert_score": 0.9},
            score=0.9,
            strategy="quality"
        )
        mock_result = RunResult(
            output="Response with reference",
            evaluation=mock_evaluation,
            model="small",
            retrieval="rag",
            latency=1.0,
            cost=0.02,
            context_used=True,
            rag_context={}
        )
        mock_process.return_value = mock_result

        request_data = {
            "input": "Test prompt",
            "reference": "Expected output"
        }
        response = self.client.post("/run-task", json=request_data)

        self.assertEqual(response.status_code, 200)
        mock_process.assert_called_once_with(request_data)

    @patch('backend.api.run_task.orchestrator.process_task')
    def test_run_task_empty_request(self, mock_process):
        mock_result = MagicMock()
        mock_process.return_value = mock_result

        response = self.client.post("/run-task", json={})

        self.assertEqual(response.status_code, 200)
        mock_process.assert_called_once_with({})

    def test_run_task_invalid_json(self):
        response = self.client.post(
            "/run-task",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 422)  # FastAPI validation error

    @patch('backend.api.compare.orchestrator.process_task')
    @patch('backend.api.compare.comparator.compare')
    def test_compare_basic(self, mock_compare, mock_process):
        from backend.core.types import RunResult, EvaluationResult

        mock_evaluation_a = EvaluationResult(
            metrics={"bert_score": 0.8},
            score=0.8,
            strategy="quality"
        )
        mock_result_a = RunResult(
            output="Response A",
            evaluation=mock_evaluation_a,
            model="small",
            retrieval="rag",
            latency=1.0,
            cost=0.01,
            context_used=True,
            rag_context={}
        )

        mock_evaluation_b = EvaluationResult(
            metrics={"bert_score": 0.7},
            score=0.7,
            strategy="quality"
        )
        mock_result_b = RunResult(
            output="Response B",
            evaluation=mock_evaluation_b,
            model="small",
            retrieval="rag",
            latency=1.2,
            cost=0.02,
            context_used=True,
            rag_context={}
        )

        mock_process.side_effect = [mock_result_a, mock_result_b]

        mock_comparison = {"winner": "A", "score_breakdown": {"A": 0.8, "B": 0.7}}
        mock_compare.return_value = mock_comparison

        request_data = {"input": "Compare this"}
        response = self.client.post("/compare", json=request_data)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("A", data)
        self.assertIn("B", data)
        self.assertIn("comparison", data)
        self.assertEqual(data["comparison"]["winner"], "A")

        # Verify orchestrator calls
        self.assertEqual(mock_process.call_count, 2)
        calls = mock_process.call_args_list
        self.assertEqual(calls[0][0][0]["input"], "Compare this")
        self.assertEqual(calls[1][0][0]["input"], "Compare this")

    @patch('backend.api.compare.orchestrator.process_task')
    @patch('backend.api.compare.comparator.compare')
    def test_compare_with_reference(self, mock_compare, mock_process):
        from backend.core.types import RunResult, EvaluationResult

        mock_result_a = RunResult(
            output="Response A",
            evaluation=EvaluationResult(metrics={}, score=0.8, strategy="quality"),
            model="small", retrieval="rag", latency=1.0, cost=0.01, context_used=True, rag_context={}
        )
        mock_result_b = RunResult(
            output="Response B",
            evaluation=EvaluationResult(metrics={}, score=0.7, strategy="quality"),
            model="small", retrieval="rag", latency=1.0, cost=0.01, context_used=True, rag_context={}
        )
        mock_process.side_effect = [mock_result_a, mock_result_b]

        mock_compare.return_value = {"winner": "B"}

        request_data = {
            "input": "Test input",
            "reference": "Expected reference"
        }
        response = self.client.post("/compare", json=request_data)

        self.assertEqual(response.status_code, 200)

        # Verify reference is passed to both calls
        calls = mock_process.call_args_list
        for call in calls:
            self.assertEqual(call[0][0]["input"], "Test input")
            self.assertEqual(call[0][0]["reference"], "Expected reference")

    def test_compare_missing_input(self):
        response = self.client.post("/compare", json={})
        # Should fail due to missing input
        self.assertEqual(response.status_code, 500)  # Internal server error from orchestrator

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @patch('backend.api.run_task.orchestrator.process_task')
    def test_run_task_orchestrator_error(self, mock_process):
        mock_process.side_effect = Exception("Orchestrator failed")

        response = self.client.post("/run-task", json={"input": "test"})

        self.assertEqual(response.status_code, 500)

    @patch('backend.api.compare.orchestrator.process_task')
    @patch('backend.api.compare.comparator.compare')
    def test_compare_orchestrator_error(self, mock_compare, mock_process):
        mock_process.side_effect = Exception("Orchestrator failed")

        response = self.client.post("/compare", json={"input": "test"})

        self.assertEqual(response.status_code, 500)


if __name__ == '__main__':
    unittest.main()