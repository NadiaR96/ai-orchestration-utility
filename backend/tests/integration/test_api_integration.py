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
    def test_run_task_with_reference_threads_dict_task(self, mock_process):
        mock_process.return_value = self._make_bundle()

        response = self.client.post(
            "/run-task",
            json={"input": "What is RAG?", "reference": "RAG stands for Retrieval Augmented Generation."}
        )

        self.assertEqual(response.status_code, 200)
        call_kwargs = mock_process.call_args
        task_arg = call_kwargs.kwargs.get("task") or call_kwargs.args[0]
        # reference must be passed as a dict so orchestrator threads it into evaluator
        self.assertIsInstance(task_arg, dict)
        self.assertEqual(task_arg["input"], "What is RAG?")
        self.assertEqual(task_arg["reference"], "RAG stands for Retrieval Augmented Generation.")

    @patch('backend.api.run.orchestrator.process_task')
    def test_run_task_without_reference_passes_string_task(self, mock_process):
        mock_process.return_value = self._make_bundle()

        response = self.client.post("/run-task", json={"input": "What is RAG?"})

        self.assertEqual(response.status_code, 200)
        call_kwargs = mock_process.call_args
        task_arg = call_kwargs.kwargs.get("task") or call_kwargs.args[0]
        # no reference → plain string task (backward compatible)
        self.assertIsInstance(task_arg, str)

    @patch('backend.api.run.orchestrator.process_task')
    def test_run_task_with_retrieval_none_threads_to_orchestrator(self, mock_process):
        mock_process.return_value = self._make_bundle()

        response = self.client.post(
            "/run-task",
            json={"input": "Test prompt", "retrieval": "none"}
        )

        self.assertEqual(response.status_code, 200)
        call_kwargs = mock_process.call_args
        self.assertEqual(call_kwargs.kwargs.get("retrieval"), "none")

    @patch('backend.api.run.orchestrator.process_task')
    def test_run_task_with_multiple_references_threads_list(self, mock_process):
        mock_process.return_value = self._make_bundle()

        response = self.client.post(
            "/run-task",
            json={
                "input": "What is RAG?",
                "reference": [
                    "RAG combines retrieval and generation.",
                    "Retrieval augmented generation grounds outputs in retrieved context.",
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        call_kwargs = mock_process.call_args
        task_arg = call_kwargs.kwargs.get("task") or call_kwargs.args[0]
        self.assertIsInstance(task_arg, dict)
        self.assertIsInstance(task_arg.get("reference"), list)
        self.assertEqual(len(task_arg["reference"]), 2)

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

    @patch('backend.api.compare.Comparator')
    @patch('backend.api.compare.Orchestrator')
    def test_compare_with_reference_threads_dict_task(self, mock_orch_cls, mock_comp_cls):
        bundle_a = self._make_bundle(output="Response A", model="small", score=0.8)
        bundle_b = self._make_bundle(output="Response B", model="large", score=0.9)
        mock_orch_cls.return_value.process_task.side_effect = [bundle_a, bundle_b]
        mock_comp_cls.return_value.compare_many.return_value = ComparisonResult(
            winner="large",
            ranking=["large", "small"],
            score_breakdown={"small": 0.8, "large": 0.9},
            strategy="balanced"
        )

        response = self.client.post(
            "/compare",
            json={
                "input": "Explain RAG",
                "models": ["small", "large"],
                "reference": "RAG stands for Retrieval Augmented Generation.",
            }
        )

        self.assertEqual(response.status_code, 200)
        # Both models should have been called with the same dict task (containing reference)
        calls = mock_orch_cls.return_value.process_task.call_args_list
        self.assertEqual(len(calls), 2)
        for call in calls:
            task_arg = call.kwargs.get("task") or call.args[0]
            self.assertIsInstance(task_arg, dict)
            self.assertEqual(task_arg["reference"], "RAG stands for Retrieval Augmented Generation.")

    @patch('backend.api.compare.Comparator')
    @patch('backend.api.compare.Orchestrator')
    def test_compare_with_retrieval_none_threads_to_orchestrator(self, mock_orch_cls, mock_comp_cls):
        bundle_a = self._make_bundle(output="Response A", model="small", score=0.8)
        bundle_b = self._make_bundle(output="Response B", model="large", score=0.9)
        mock_orch_cls.return_value.process_task.side_effect = [bundle_a, bundle_b]
        mock_comp_cls.return_value.compare_many.return_value = ComparisonResult(
            winner="large",
            ranking=["large", "small"],
            score_breakdown={"small": 0.8, "large": 0.9},
            strategy="balanced",
        )

        response = self.client.post(
            "/compare",
            json={"input": "Compare this", "models": ["small", "large"], "retrieval": "none"}
        )

        self.assertEqual(response.status_code, 200)
        calls = mock_orch_cls.return_value.process_task.call_args_list
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertEqual(call.kwargs.get("retrieval"), "none")

    @patch('backend.api.compare.Comparator')
    @patch('backend.api.compare.Orchestrator')
    def test_compare_with_multiple_references_threads_list(self, mock_orch_cls, mock_comp_cls):
        bundle_a = self._make_bundle(output="Response A", model="small", score=0.8)
        bundle_b = self._make_bundle(output="Response B", model="large", score=0.9)
        mock_orch_cls.return_value.process_task.side_effect = [bundle_a, bundle_b]
        mock_comp_cls.return_value.compare_many.return_value = ComparisonResult(
            winner="large",
            ranking=["large", "small"],
            score_breakdown={"small": 0.8, "large": 0.9},
            strategy="balanced",
        )

        response = self.client.post(
            "/compare",
            json={
                "input": "Explain RAG",
                "models": ["small", "large"],
                "reference": [
                    "RAG combines retrieval and generation.",
                    "Retrieval augmented generation grounds outputs in retrieved context.",
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        calls = mock_orch_cls.return_value.process_task.call_args_list
        self.assertEqual(len(calls), 2)
        for call in calls:
            task_arg = call.kwargs.get("task") or call.args[0]
            self.assertIsInstance(task_arg.get("reference"), list)

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

    @patch('backend.api.leaderboard.orchestrator.process_task')
    def test_leaderboard_prompt_mode_basic(self, mock_process):
        mock_process.side_effect = [
            self._make_bundle(output="Small", model="small", score=0.75, strategy="balanced"),
            self._make_bundle(output="Large", model="large", score=0.92, strategy="balanced"),
        ]

        response = self.client.post(
            "/leaderboard",
            json={
                "input": "Rank these",
                "models": ["small", "large"],
                "sort_strategy": "balanced",
                "page": 1,
                "page_size": 10,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "prompt")
        self.assertEqual(data["total_items"], 2)
        self.assertFalse(data["has_more"])
        self.assertEqual(data["items"][0]["model"], "large")
        self.assertIn("balanced", data["strategy_rankings"])
        self.assertIn("quality", data["strategy_rankings"])
        self.assertEqual(mock_process.call_count, 2)

    @patch('backend.api.leaderboard.orchestrator.process_task')
    def test_leaderboard_prompt_mode_pagination(self, mock_process):
        mock_process.side_effect = [
            self._make_bundle(output="Small", model="small", score=0.7, strategy="balanced"),
            self._make_bundle(output="Medium", model="medium", score=0.8, strategy="balanced"),
            self._make_bundle(output="Large", model="large", score=0.9, strategy="balanced"),
        ]

        response = self.client.post(
            "/leaderboard",
            json={
                "input": "Paginate",
                "models": ["small", "medium", "large"],
                "sort_strategy": "balanced",
                "page": 2,
                "page_size": 1,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["page"], 2)
        self.assertEqual(data["page_size"], 1)
        self.assertEqual(data["total_items"], 3)
        self.assertTrue(data["has_more"])
        self.assertEqual(data["next_page"], 3)
        self.assertEqual(len(data["items"]), 1)

    @patch('backend.api.leaderboard._load_latest_entries_from_logs')
    def test_leaderboard_historical_mode_basic(self, mock_load):
        run_small = RunResult(
            output="Small",
            model="small",
            retrieval="historical",
            latency=1.2,
            cost=0.01,
            context_used=False,
            rag_context={},
        )
        run_large = RunResult(
            output="Large",
            model="large",
            retrieval="historical",
            latency=1.0,
            cost=0.02,
            context_used=False,
            rag_context={},
        )
        eval_small = EvaluationResult(metrics={"bert_score": 0.6}, score=0.6, strategy="balanced")
        eval_large = EvaluationResult(metrics={"bert_score": 0.9}, score=0.9, strategy="balanced")

        from backend.core.types import LeaderboardEntry

        mock_load.return_value = [
            LeaderboardEntry(
                model="small",
                run=run_small,
                evaluation=eval_small,
                scores_by_strategy={"balanced": 0.6, "quality": 0.6, "cost_aware": 0.6, "rag": 0.6},
                ranks_by_strategy={},
                narrative="",
            ),
            LeaderboardEntry(
                model="large",
                run=run_large,
                evaluation=eval_large,
                scores_by_strategy={"balanced": 0.9, "quality": 0.9, "cost_aware": 0.9, "rag": 0.9},
                ranks_by_strategy={},
                narrative="",
            ),
        ]

        response = self.client.get("/leaderboard?page=1&page_size=10&sort_strategy=balanced")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "historical")
        self.assertEqual(data["items"][0]["model"], "large")
        self.assertEqual(data["total_items"], 2)
        mock_load.assert_called_once()

    @patch('backend.api.leaderboard._load_latest_entries_from_logs')
    def test_leaderboard_experiments_mode_basic(self, mock_load):
        from backend.core.types import LeaderboardEntry

        run = RunResult(
            output="Experiment output",
            model="quality",
            retrieval="historical",
            latency=1.1,
            cost=0.02,
            context_used=False,
            rag_context={},
        )
        evaluation = EvaluationResult(metrics={"bert_score": 0.88}, score=0.88, strategy="balanced")

        mock_load.return_value = [
            LeaderboardEntry(
                model="quality",
                run=run,
                evaluation=evaluation,
                scores_by_strategy={"balanced": 0.88, "quality": 0.88, "cost_aware": 0.88, "rag": 0.88},
                ranks_by_strategy={},
                narrative="",
            )
        ]

        response = self.client.get("/leaderboard/experiments?page=1&page_size=10&sort_strategy=balanced")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "experiments")
        self.assertEqual(data["total_items"], 1)
        self.assertEqual(data["items"][0]["model"], "quality")

    @patch('backend.api.leaderboard._load_latest_entries_from_logs')
    def test_leaderboard_live_mode_basic(self, mock_load):
        from backend.core.types import LeaderboardEntry

        run = RunResult(
            output="Live output",
            model="small",
            retrieval="rag",
            latency=0.9,
            cost=0.01,
            context_used=True,
            rag_context={},
        )
        evaluation = EvaluationResult(metrics={"bert_score": 0.8}, score=0.8, strategy="balanced")

        mock_load.return_value = [
            LeaderboardEntry(
                model="small",
                run=run,
                evaluation=evaluation,
                scores_by_strategy={"balanced": 0.8, "quality": 0.8, "cost_aware": 0.8, "rag": 0.8},
                ranks_by_strategy={},
                narrative="",
            )
        ]

        response = self.client.get("/leaderboard/live?page=1&page_size=10&sort_strategy=balanced&window_hours=24&min_samples=1")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "live")
        self.assertEqual(data["total_items"], 1)
        self.assertEqual(data["items"][0]["model"], "small")
        self.assertIn("trend", data["items"][0])
        self.assertIn("direction", data["items"][0]["trend"])

    @patch('backend.api.leaderboard._load_latest_entries_from_logs')
    def test_leaderboard_live_window_avg_ranking(self, mock_load):
        from backend.core.types import LeaderboardEntry

        run_consistent = RunResult(
            output="Good output",
            model="consistent",
            retrieval="rag",
            latency=1.0,
            cost=0.01,
            context_used=True,
            rag_context={},
        )
        run_flashy = RunResult(
            output="Great output this time",
            model="flashy",
            retrieval="rag",
            latency=1.0,
            cost=0.01,
            context_used=True,
            rag_context={},
        )
        eval_consistent = EvaluationResult(metrics={"bert_score": 0.7}, score=0.7, strategy="balanced")
        eval_flashy = EvaluationResult(metrics={"bert_score": 0.95}, score=0.95, strategy="balanced")

        entry_consistent = LeaderboardEntry(
            model="consistent",
            run=run_consistent,
            evaluation=eval_consistent,
            scores_by_strategy={"balanced": 0.7, "quality": 0.7, "cost_aware": 0.7, "rag": 0.7},
            ranks_by_strategy={},
            narrative="",
            latest_score=0.7,
            sample_count=5,
        )
        entry_consistent._window_avg_score = 0.9  # type: ignore[attr-defined]

        entry_flashy = LeaderboardEntry(
            model="flashy",
            run=run_flashy,
            evaluation=eval_flashy,
            scores_by_strategy={"balanced": 0.95, "quality": 0.95, "cost_aware": 0.95, "rag": 0.95},
            ranks_by_strategy={},
            narrative="",
            latest_score=0.95,
            sample_count=5,
        )
        entry_flashy._window_avg_score = 0.5  # type: ignore[attr-defined]

        mock_load.return_value = [entry_consistent, entry_flashy]

        response = self.client.get(
            "/leaderboard/live?page=1&page_size=10&sort_strategy=balanced&ranking_basis=window_avg"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["mode"], "live")
        self.assertEqual(data["items"][0]["model"], "consistent",
                         "consistent should rank first (higher window avg despite lower latest score)")

    @patch('backend.api.leaderboard._load_latest_entries_from_logs')
    def test_leaderboard_live_latest_ranking_uses_snapshot(self, mock_load):
        from backend.core.types import LeaderboardEntry

        run_a = RunResult(output="A", model="a", retrieval="rag", latency=1.0, cost=0.01, context_used=True, rag_context={})
        run_b = RunResult(output="B", model="b", retrieval="rag", latency=1.0, cost=0.01, context_used=True, rag_context={})
        eval_a = EvaluationResult(metrics={"bert_score": 0.9}, score=0.9, strategy="balanced")
        eval_b = EvaluationResult(metrics={"bert_score": 0.5}, score=0.5, strategy="balanced")

        entry_a = LeaderboardEntry(
            model="a", run=run_a, evaluation=eval_a,
            scores_by_strategy={"balanced": 0.9, "quality": 0.9, "cost_aware": 0.9, "rag": 0.9},
            ranks_by_strategy={}, narrative="", latest_score=0.9, sample_count=1,
        )
        entry_b = LeaderboardEntry(
            model="b", run=run_b, evaluation=eval_b,
            scores_by_strategy={"balanced": 0.5, "quality": 0.5, "cost_aware": 0.5, "rag": 0.5},
            ranks_by_strategy={}, narrative="", latest_score=0.5, sample_count=1,
        )
        mock_load.return_value = [entry_a, entry_b]

        response = self.client.get(
            "/leaderboard/live?page=1&page_size=10&sort_strategy=balanced&ranking_basis=latest"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["items"][0]["model"], "a",
                         "ranking_basis=latest should rank by snapshot scores")

    def test_leaderboard_live_invalid_ranking_basis(self):
        response = self.client.get("/leaderboard/live?ranking_basis=mean")
        self.assertEqual(response.status_code, 422)

    def test_leaderboard_live_invalid_window(self):
        response = self.client.get("/leaderboard/live?page=1&page_size=10&window_hours=0")
        self.assertEqual(response.status_code, 422)

    def test_leaderboard_invalid_page(self):
        response = self.client.get("/leaderboard?page=0&page_size=10")
        self.assertEqual(response.status_code, 422)

    def test_leaderboard_invalid_aggregation(self):
        response = self.client.get("/leaderboard?page=1&page_size=10&aggregation=mean")
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
