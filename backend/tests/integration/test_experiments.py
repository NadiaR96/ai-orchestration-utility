import unittest
import tempfile
import os
import json
import shutil
from unittest.mock import patch, MagicMock
from backend.experiments.runner import ExperimentRunner
from backend.experiments.tracker import ExperimentTracker
from backend.experiments.results import ExperimentResult


class TestExperimentWorkflow(unittest.TestCase):
    def setUp(self):
        self.runner = ExperimentRunner()

    @patch('backend.orchestrator.Orchestrator.process_task')
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
            metrics=["bert_score"]
        )

    @patch('backend.orchestrator.Orchestrator.process_task')
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
            metrics=None
        )

    @patch('backend.orchestrator.Orchestrator.process_task')
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

    @patch('backend.orchestrator.Orchestrator.process_task')
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

    @patch('backend.orchestrator.Orchestrator.process_task')
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


class TestExperimentTracker(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test_logs.jsonl")
        self.tracker = ExperimentTracker(self.log_file)

    def tearDown(self):
        # Clean up temp files and directories recursively
        import shutil
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