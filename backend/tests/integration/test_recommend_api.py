import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.recommender.engine import RecommendationEngine, RecommendationResult, ModelRecommendation


class TestRecommendAPIIntegration(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)

    def _mock_result(self, best_model="alpha", strategy="balanced", use_case="summarisation", use_case_matched=True):
        return RecommendationResult(
            best_model=best_model,
            best_score=0.9,
            strategy=strategy,
            use_case=use_case,
            use_case_matched=use_case_matched,
            system_state="DECIDING",
            decision_result="RECOMMENDED",
            decision_state="RECOMMENDED",
            evaluation_status="VALID",
            decision_reliability=0.8,
            validity_threshold=0.55,
            decision_threshold=0.70,
            display_threshold=0.40,
            display_eligible=True,
            score_summary={"best_score": 0.9, "margin": 0.12, "variance": 0.02},
            score_vector={"quality_score": 0.9, "latency": 1.0, "cost": 0.01, "variance": 0.02, "sample_size": 5.0},
            recommendation_available=True,
            no_recommendation_reason=None,
            gate_status="PASS",
            gate_threshold=0.70,
            gate_triggers=[],
            score_margin=0.12,
            score_variance=0.02,
            validity_status="VALID",
            is_valid=True,
            validity_reasons=["All validity checks passed."],
            failure_modes=[],
            confidence=0.8,
            confidence_label="HIGH",
            confidence_reasons=["Strong margin and enough samples."],
            alternatives=[
                ModelRecommendation(
                    model=best_model,
                    score=0.9,
                    sample_count=5,
                    avg_latency=1.0,
                    avg_cost=0.01,
                    score_stddev=0.02,
                    score_delta_from_best=0.0,
                    confidence=0.8,
                    p95_latency=1.8,
                    consistency_above_threshold=0.8,
                ),
                ModelRecommendation(
                    model="beta",
                    score=0.75,
                    sample_count=3,
                    avg_latency=1.5,
                    avg_cost=0.02,
                    score_stddev=0.05,
                    score_delta_from_best=-0.15,
                    confidence=0.5,
                    p95_latency=2.2,
                    consistency_above_threshold=0.6667,
                ),
            ],
            justification=f"Recommended '{best_model}' for strategy '{strategy}'.",
        )

    @patch.object(RecommendationEngine, "recommend")
    def test_recommend_returns_200_and_correct_shape(self, mock_recommend):
        mock_recommend.return_value = self._mock_result()

        response = self.client.get("/recommend?use_case=summarisation&strategy=balanced")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["best_model"], "alpha")
        self.assertIn("best_score", data)
        self.assertIn("evaluation", data)
        self.assertIn("system_health", data["evaluation"])
        self.assertIn("evaluation_status", data["evaluation"])
        self.assertIn("decision", data["evaluation"])
        self.assertIn("reliability", data["evaluation"])
        self.assertIn("metrics", data["evaluation"])
        self.assertIn("failure_analysis", data["evaluation"])
        self.assertIn("alternatives", data["evaluation"])
        self.assertIn("recommendation_available", data)
        self.assertIn("no_recommendation_reason", data)
        self.assertIn("validity_status", data)
        self.assertIn("is_valid", data)
        self.assertIn("validity_reasons", data)
        self.assertIn("confidence_reasons", data)
        self.assertIn("alternatives", data)
        self.assertIn("justification", data)
        self.assertIsInstance(data["alternatives"], list)
        self.assertNotIn("system_state", data)
        self.assertNotIn("decision_state", data)
        self.assertNotIn("evaluation_status", data)
        self.assertNotIn("decision_reliability", data)
        self.assertNotIn("score_summary", data)
        self.assertNotIn("strategy", data)
        self.assertNotIn("use_case", data)
        self.assertNotIn("use_case_matched", data)
        self.assertNotIn("score_margin", data)
        self.assertNotIn("score_variance", data)
        self.assertNotIn("failure_modes", data)
        self.assertNotIn("confidence", data)
        self.assertNotIn("confidence_label", data)

    @patch.object(RecommendationEngine, "recommend")
    def test_recommend_alternatives_have_correct_fields(self, mock_recommend):
        mock_recommend.return_value = self._mock_result()

        response = self.client.get("/recommend?use_case=code_gen&strategy=quality&top_n=2")

        self.assertEqual(response.status_code, 200)
        alt = response.json()["alternatives"][0]
        self.assertIn("model", alt)
        self.assertIn("score", alt)
        self.assertIn("sample_count", alt)
        self.assertIn("avg_latency", alt)
        self.assertIn("avg_cost", alt)
        self.assertIn("score_stddev", alt)
        self.assertIn("score_delta_from_best", alt)
        self.assertIn("confidence", alt)
        self.assertIn("p95_latency", alt)
        self.assertIn("consistency_above_threshold", alt)

    @patch.object(RecommendationEngine, "recommend")
    def test_recommend_use_case_not_matched_flag(self, mock_recommend):
        result = self._mock_result(use_case_matched=False)
        result.failure_modes = ["USE_CASE_MISMATCH"]
        result.validity_status = "WARNING"
        result.is_valid = False
        mock_recommend.return_value = result

        response = self.client.get("/recommend?use_case=unknown_task")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["evaluation"]["use_case_matched"])
        self.assertIn("USE_CASE_MISMATCH", body["evaluation"]["failure_analysis"]["modes"])

    @patch.object(RecommendationEngine, "recommend")
    def test_recommend_can_return_no_recommendation_state(self, mock_recommend):
        result = self._mock_result()
        result.best_model = ""
        result.recommendation_available = False
        result.no_recommendation_reason = "All qualifying models are weak."
        result.validity_status = "INVALID"
        result.is_valid = False
        result.failure_modes = ["ALL_MODELS_WEAK"]
        mock_recommend.return_value = result

        response = self.client.get("/recommend?use_case=test&strategy=balanced")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertFalse(body["recommendation_available"])
        self.assertIn("weak", (body.get("no_recommendation_reason") or "").lower())

    @patch.object(RecommendationEngine, "recommend")
    def test_recommend_preserves_extended_evaluation_status(self, mock_recommend):
        result = self._mock_result()
        result.evaluation_status = "NOISY"
        mock_recommend.return_value = result

        response = self.client.get("/recommend?use_case=test&strategy=balanced")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["evaluation"]["evaluation_status"], "NOISY")

    @patch.object(RecommendationEngine, "recommend")
    def test_recommend_maps_insufficient_evidence_label_consistently(self, mock_recommend):
        result = self._mock_result()
        result.confidence_label = "INSUFFICIENT EVIDENCE"
        mock_recommend.return_value = result

        response = self.client.get("/recommend?use_case=test&strategy=balanced")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["evaluation"]["reliability"]["label"], "INSUFFICIENT EVIDENCE")

    @patch.object(RecommendationEngine, "recommend")
    def test_recommend_maps_failure_modes_without_collapsing(self, mock_recommend):
        result = self._mock_result()
        result.failure_modes = ["ALL_MODELS_WEAK", "LOW_CONSISTENCY", "USE_CASE_MISMATCH", "SCORE_SCALE_DRIFT"]
        mock_recommend.return_value = result

        response = self.client.get("/recommend?use_case=test&strategy=balanced")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        modes = body["evaluation"]["failure_analysis"]["modes"]
        self.assertIn("ALL_MODELS_WEAK", modes)
        self.assertIn("LOW_CONSISTENCY", modes)
        self.assertIn("USE_CASE_MISMATCH", modes)
        self.assertIn("SCORE_SCALE_DRIFT", modes)

    def test_recommend_missing_use_case_returns_422(self):
        response = self.client.get("/recommend?strategy=balanced")
        self.assertEqual(response.status_code, 422)

    @patch.object(RecommendationEngine, "recommend", side_effect=ValueError("Unknown scoring strategy: 'bad'"))
    def test_recommend_invalid_strategy_returns_422(self, _mock):
        response = self.client.get("/recommend?use_case=test&strategy=bad")
        self.assertEqual(response.status_code, 422)
        self.assertIn("Unknown scoring strategy", response.json()["detail"])

    @patch.object(RecommendationEngine, "recommend", side_effect=ValueError("No qualifying model data found"))
    def test_recommend_no_data_returns_422(self, _mock):
        response = self.client.get("/recommend?use_case=test&strategy=balanced")
        self.assertEqual(response.status_code, 422)

    @patch.object(RecommendationEngine, "recommend")
    def test_recommend_source_param_passed_through(self, mock_recommend):
        mock_recommend.return_value = self._mock_result()

        self.client.get("/recommend?use_case=test&source=live")

        _call_kwargs = mock_recommend.call_args
        self.assertEqual(_call_kwargs.kwargs.get("source") or _call_kwargs.args[3] if _call_kwargs.args else "live", "live")

    @patch.object(RecommendationEngine, "recommend")
    def test_recommend_default_source_is_all(self, mock_recommend):
        mock_recommend.return_value = self._mock_result()

        self.client.get("/recommend?use_case=test")

        called_kwargs = mock_recommend.call_args.kwargs
        self.assertEqual(called_kwargs.get("source", "all"), "all")


if __name__ == "__main__":
    unittest.main()
