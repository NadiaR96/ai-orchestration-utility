import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend.main import app
from backend.core.types import RunResult, EvaluationResult, RunBundle, ComparisonResult


class TestAPIIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def _make_bundle(self, output="Generated response", model="small", score=0.85, strategy="balanced"):
        run = RunResult(
            output=output,
            model=model,
            retrieval="rag",
            latency=1.2,
            cost=0.01,
            context_used=True,
            rag_context={}
        )
        evaluation = EvaluationResult(
            metrics={"bert_score": 0.8},
            score=score,
            strategy=strategy
        )
        return RunBundle(run=run, evaluation=evaluation)

    @patch('backend.api.run.orchestrator.process_task')
    def test_run_task_string_input(self, mock_process):
        mock_process.return_value = self._make_bundle()

        response = self.client.post("/run-task", json={"input": "Test prompt"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["run"]["output"], "Generated response")
        self.assertEqual(data["evaluations"]["single"]["score"], 0.85)
        self.assertIsNone(data["comparison"])
        mock_process.assert_called_once()

    @patch('backend.api.run.orchestrator.process_task')
    def test_run_task_with_model(self, mock_process):
        mock_process.return_value = self._make_bundle(model="large")

        response = self.client.post("/run-task", json={"input": "Test prompt", "model": "large"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["run"]["model"], "large")
        mock_process.assert_called_once()

    @patch('backend.api.run.orchestrator.process_task')
    def test_run_task_missing_input(self, mock_process):
        response = self.client.post("/run-task", json={})
        self.assertEqual(response.status_code, 422)
        mock_process.assert_not_called()

    def test_run_task_invalid_json(self):
        response = self.client.post(
            "/run-task",
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 422)

    @patch('backend.api.compare.Comparator')
    @patch('backend.api.compare.Orchestrator')
    def test_compare_basic(self, mock_orch_cls, mock_comp_cls):
        bundle_a = self._make_bundle(output="Response A", model="small", score=0.8)
        bundle_b = self._make_bundle(output="Response B", model="large", score=0.9)
        mock_orch_cls.return_value.process_task.side_effect = [bundle_a, bundle_b]
        mock_comp_cls.return_value.compare_many.return_value = ComparisonResult(
            winner="large",
            ranking=["large", "small"],
            score_breakdown={"small": 0.8, "large": 0.9},
            strategy="balanced"
        )

        response = self.client.post("/compare", json={"input": "Compare this", "models": ["small", "large"]})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("small", data["runs"])
        self.assertIn("large", data["runs"])
        self.assertEqual(data["comparison"]["winner"], "large")

    @patch('backend.api.compare.Comparator')
    @patch('backend.api.compare.Orchestrator')
    def test_compare_single_model(self, mock_orch_cls, mock_comp_cls):
        bundle = self._make_bundle(output="Response A", model="small", score=0.8)
        mock_orch_cls.return_value.process_task.return_value = bundle

        response = self.client.post("/compare", json={"input": "Test input"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        mock_comp_cls.return_value.compare_many.assert_not_called()
        self.assertIsNone(data["comparison"])

    @patch('backend.api.compare.Comparator')
    @patch('backend.api.compare.Orchestrator')
    def test_compare_with_strategy(self, mock_orch_cls, mock_comp_cls):
        bundle_a = self._make_bundle(model="small", score=0.7, strategy="cost_aware")
        bundle_b = self._make_bundle(model="large", score=0.9, strategy="cost_aware")
        mock_orch_cls.return_value.process_task.side_effect = [bundle_a, bundle_b]
        mock_comp_cls.return_value.compare_many.return_value = ComparisonResult(
            winner="large",
            ranking=["large", "small"],
            score_breakdown={"small": 0.7, "large": 0.9},
            strategy="cost_aware"
        )

        response = self.client.post(
            "/compare",
            json={"input": "Test input", "models": ["small", "large"], "strategy": "cost_aware"}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["comparison"]["strategy"], "cost_aware")

    def test_compare_missing_input(self):
        response = self.client.post("/compare", json={})
        self.assertEqual(response.status_code, 422)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    @patch('backend.api.run.orchestrator.process_task')
    def test_run_task_orchestrator_error(self, mock_process):
        mock_process.side_effect = Exception("Orchestrator failed")

        response = self.client.post("/run-task", json={"input": "test"})
        self.assertEqual(response.status_code, 500)

    @patch('backend.api.compare.Orchestrator')
    def test_compare_orchestrator_error(self, mock_orch_cls):
        mock_orch_cls.return_value.process_task.side_effect = Exception("Orchestrator failed")

        response = self.client.post("/compare", json={"input": "test"})
        self.assertEqual(response.status_code, 500)


if __name__ == '__main__':
    unittest.main()
