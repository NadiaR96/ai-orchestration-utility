import unittest
import tempfile
import os
import json
import shutil
from unittest.mock import patch, MagicMock
from backend.experiments.runner import ExperimentRunner
from backend.experiments.tracker import ExperimentTracker
from backend.experiments.experiment import ExperimentConfig
from backend.experiments.results import ExperimentResult
from backend.core.types import RunResult, EvaluationResult, RunBundle


class TestExperimentWorkflow(unittest.TestCase):
    def setUp(self):
        self.runner = ExperimentRunner()

    def _make_bundle(
        self,
        model: str,
        score: float,
        latency: float,
        cost: float,
        prompt_tokens: int = 20,
        output_tokens: int = 10,
    ) -> RunBundle:
        total_tokens = prompt_tokens + output_tokens
        return RunBundle(
            run=RunResult(
                output=f"Output for {model} @ {score}",
                model=model,
                retrieval="rag",
                latency=latency,
                cost=cost,
                context_used=True,
                rag_context={},
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_per_1k_tokens=(cost / max(float(total_tokens), 1.0)) * 1000.0,
            ),
            evaluation=EvaluationResult(
                metrics={"bert_score": score, "latency": latency, "cost": cost},
                score=score,
                strategy="balanced",
            ),
        )

    @patch('backend.experiments.runner.Orchestrator.process_task')
    def test_run_single_experiment(self, mock_process):
        # Mock the orchestrator response
        mock_result = {
            "output": "Generated response",
            "model": "small",
            "retrieval": "rag",
            "evaluation": {"score": 0.85, "metrics": {"bert_score": 0.8}},
            "rag_context": {"chunks": ["context chunk"]}
        }
        mock_process.return_value = mock_result

        config = {
            "model": "small",
            "retrieval": "rag",
            "metrics": ["bert_score"]
        }

        result = self.runner.run_single("Test prompt", config)

        self.assertIsInstance(result, ExperimentResult)
        self.assertEqual(result.output, "Generated response")
        self.assertEqual(result.model, "small")
        self.assertEqual(result.retrieval, "rag")
        self.assertEqual(result.metrics["score"], 0.85)

        mock_process.assert_called_once_with(
            task={"input": "Test prompt", "reference": None},
            model="small",
            retrieval="rag",
        )

    @patch('backend.experiments.runner.Orchestrator.process_task')
    def test_run_single_with_reference(self, mock_process):
        mock_result = {
            "output": "Response with reference",
            "model": "large",
            "retrieval": "none",
            "evaluation": {"score": 0.9},
            "rag_context": {}
        }
        mock_process.return_value = mock_result

        config = {"model": "large", "retrieval": "none"}
        result = self.runner.run_single("Test prompt", config, reference="Expected output")

        self.assertEqual(result.output, "Response with reference")
        mock_process.assert_called_once_with(
            task={"input": "Test prompt", "reference": "Expected output"},
            model="large",
            retrieval="none",
        )

    @patch('backend.experiments.runner.Orchestrator.process_task')
    def test_run_batch_experiments(self, mock_process):
        # Mock different results for each config
        mock_results = [
            {
                "output": "Response 1",
                "model": "small",
                "retrieval": "rag",
                "evaluation": {"score": 0.8},
                "rag_context": {"chunks": ["chunk1"]}
            },
            {
                "output": "Response 2",
                "model": "large",
                "retrieval": "none",
                "evaluation": {"score": 0.9},
                "rag_context": {}
            }
        ]
        mock_process.side_effect = mock_results

        configs = [
            {"model": "small", "retrieval": "rag"},
            {"model": "large", "retrieval": "none"}
        ]

        results = self.runner.run_batch("Batch prompt", configs)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].output, "Response 1")
        self.assertEqual(results[1].output, "Response 2")
        self.assertEqual(mock_process.call_count, 2)

    @patch('backend.experiments.runner.Orchestrator.process_task')
    def test_compare_pair_experiments(self, mock_process):
        mock_results = [
            {
                "output": "Response A",
                "model": "small",
                "retrieval": "rag",
                "evaluation": {"score": 0.8},
                "rag_context": {"chunks": ["chunk A"]}
            },
            {
                "output": "Response B",
                "model": "large",
                "retrieval": "none",
                "evaluation": {"score": 0.9},
                "rag_context": {}
            }
        ]
        mock_process.side_effect = mock_results

        config_a = {"model": "small", "retrieval": "rag"}
        config_b = {"model": "large", "retrieval": "none"}

        def custom_compare(exp_a, exp_b):
            return {"winner": "B", "margin": exp_b.metrics["score"] - exp_a.metrics["score"]}

        result = self.runner.compare_pair("Compare prompt", config_a, config_b, compare_fn=custom_compare)

        self.assertIn("A", result)
        self.assertIn("B", result)
        self.assertIn("comparison", result)
        self.assertEqual(result["comparison"]["winner"], "B")
        self.assertAlmostEqual(result["comparison"]["margin"], 0.1, places=5)

    @patch('backend.experiments.runner.Orchestrator.process_task')
    def test_compare_pair_without_custom_compare(self, mock_process):
        mock_results = [
            {
                "output": "Response A",
                "model": "small",
                "retrieval": "rag",
                "evaluation": {"score": 0.7},
                "rag_context": {"chunks": []}
            },
            {
                "output": "Response B",
                "model": "large",
                "retrieval": "none",
                "evaluation": {"score": 0.8},
                "rag_context": {}
            }
        ]
        mock_process.side_effect = mock_results

        config_a = {"model": "small"}
        config_b = {"model": "large"}

        result = self.runner.compare_pair("Compare prompt", config_a, config_b)

        self.assertIsNone(result["comparison"])
        self.assertEqual(result["A"]["output"], "Response A")
        self.assertEqual(result["B"]["output"], "Response B")

    @patch("backend.experiments.runner.Orchestrator.process_task")
    def test_run_honors_runs_per_input_and_aggregates(self, mock_process):
        mock_process.side_effect = [
            self._make_bundle(model="small", score=0.60, latency=1.0, cost=0.001, prompt_tokens=10, output_tokens=5),
            self._make_bundle(model="small", score=0.90, latency=2.0, cost=0.003, prompt_tokens=30, output_tokens=15),
        ]

        config = ExperimentConfig(
            name="exp-runs-per-input",
            inputs=["Prompt A"],
            models=["small"],
            strategy="balanced",
            runs_per_input=2,
        )

        result = self.runner.run(config)

        self.assertEqual(mock_process.call_count, 2)
        aggregated = result.run_matrix["Prompt A"]["small"]
        self.assertAlmostEqual(aggregated.evaluation.score, 0.75, places=5)
        self.assertAlmostEqual(aggregated.run.latency, 1.5, places=5)
        self.assertAlmostEqual(aggregated.run.cost, 0.002, places=5)
        self.assertEqual(aggregated.run.prompt_tokens, 20)
        self.assertEqual(aggregated.run.output_tokens, 10)
        self.assertEqual(aggregated.run.total_tokens, 30)
        self.assertAlmostEqual(aggregated.run.cost_per_1k_tokens, (0.0666666667 + 0.0666666667) / 2, places=4)

    @patch("backend.experiments.runner.Orchestrator.process_task")
    def test_run_logs_each_iteration(self, mock_process):
        mock_process.side_effect = [
            self._make_bundle(model="small", score=0.7, latency=1.0, cost=0.001),
            self._make_bundle(model="small", score=0.8, latency=1.1, cost=0.0011),
            self._make_bundle(model="large", score=0.75, latency=1.2, cost=0.0012),
            self._make_bundle(model="large", score=0.85, latency=1.3, cost=0.0013),
        ]

        config = ExperimentConfig(
            name="exp-log-iterations",
            inputs=["Prompt A"],
            models=["small", "large"],
            strategy="balanced",
            runs_per_input=2,
        )

        with patch.object(self.runner.tracker, "log") as mock_log:
            self.runner.run(config)

        # 1 input x 2 models x 2 iterations
        self.assertEqual(mock_log.call_count, 4)

        logged_iterations = sorted(call.args[0]["run_iteration"] for call in mock_log.call_args_list)
        self.assertEqual(logged_iterations, [1, 1, 2, 2])
        for call in mock_log.call_args_list:
            self.assertEqual(call.kwargs["experiment_id"], "exp-log-iterations")


class TestExperimentTracker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_logs.jsonl")
        self.tracker = ExperimentTracker(self.log_file)

    def tearDown(self):
        # Clean up temp files and directories recursively
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_log_experiment_result(self):
        result = {
            "output": "Test output",
            "model": "small",
            "score": 0.85
        }

        logged = self.tracker.log(result)

        # Check that record was created with required fields
        self.assertIn("run_id", logged)
        self.assertIn("timestamp", logged)
        self.assertEqual(logged["output"], "Test output")
        self.assertEqual(logged["model"], "small")
        self.assertEqual(logged["score"], 0.85)

        # Check file was written
        self.assertTrue(os.path.exists(self.log_file))

        # Check file contents
        with open(self.log_file, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["output"], "Test output")

    def test_log_creates_directory(self):
        # Test with nested directory that doesn't exist
        nested_log_file = os.path.join(self.temp_dir, "nested", "dir", "logs.jsonl")
        tracker = ExperimentTracker(nested_log_file)

        result = {"test": "data"}
        tracker.log(result)

        self.assertTrue(os.path.exists(nested_log_file))

    def test_log_multiple_records(self):
        results = [
            {"experiment": 1, "score": 0.8},
            {"experiment": 2, "score": 0.9},
            {"experiment": 3, "score": 0.7}
        ]

        for result in results:
            self.tracker.log(result)

        # Check all records are in file
        with open(self.log_file, "r") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)

            for i, line in enumerate(lines):
                record = json.loads(line)
                self.assertEqual(record["experiment"], i + 1)


if __name__ == '__main__':
    unittest.main()