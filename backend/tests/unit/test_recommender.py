import json
import tempfile
import unittest
from pathlib import Path

from backend.recommender.engine import RecommendationEngine, _seed_strategy_scores


def _write_logs(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


class TestSeedStrategyScores(unittest.TestCase):
    def test_uses_scores_by_strategy_when_present(self):
        record = {
            "model": "m",
            "scores_by_strategy": {"balanced": 0.8, "quality": 0.7},
        }
        result = _seed_strategy_scores(record)
        self.assertAlmostEqual(result["balanced"], 0.8)
        self.assertAlmostEqual(result["quality"], 0.7)

    def test_maps_single_score_to_record_strategy(self):
        record = {"model": "m", "score": 0.6, "strategy": "quality"}
        result = _seed_strategy_scores(record)
        self.assertEqual(result, {"quality": 0.6})

    def test_falls_back_to_target_strategy_for_legacy_record(self):
        record = {"model": "m", "score": 0.6}
        result = _seed_strategy_scores(record, target_strategy="cost_aware")
        self.assertEqual(result, {"cost_aware": 0.6})

    def test_returns_empty_when_no_score(self):
        result = _seed_strategy_scores({"model": "m"})
        self.assertEqual(result, {})


class TestRecommendationEngine(unittest.TestCase):
    def _engine(self, records: list) -> RecommendationEngine:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w", encoding="utf-8")
        for r in records:
            tmp.write(json.dumps(r) + "\n")
        tmp.close()
        engine = RecommendationEngine(log_path=Path(tmp.name))
        # Redirect persistence to a temp file too
        engine._persist = lambda result, source: None  # type: ignore[method-assign]
        return engine

    def _record(self, model, score, source="experiment", use_case=None, latency=1.0, cost=0.01):
        r = {
            "model": model,
            "score": score,
            "source": source,
            "latency": latency,
            "cost": cost,
        }
        if use_case:
            r["use_case"] = use_case
        return r

    def test_basic_recommendation_returns_best_model(self):
        engine = self._engine([
            self._record("alpha", 0.9),
            self._record("alpha", 0.88),
            self._record("alpha", 0.92),
            self._record("beta", 0.75),
            self._record("beta", 0.74),
            self._record("beta", 0.76),
        ])
        result = engine.recommend("summarisation", "balanced")
        self.assertEqual(result.best_model, "alpha")
        self.assertAlmostEqual(result.best_score, 0.9, places=1)

    def test_ranked_alternatives_ordered_by_score(self):
        engine = self._engine([
            self._record("alpha", 0.9),
            self._record("beta", 0.6),
            self._record("gamma", 0.75),
        ])
        result = engine.recommend("any", "balanced", top_n=3)
        scores = [a.score for a in result.alternatives]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_use_case_tag_match_filters_records(self):
        engine = self._engine([
            self._record("alpha", 0.9, use_case="summarisation"),
            self._record("alpha", 0.88, use_case="summarisation"),
            self._record("alpha", 0.92, use_case="summarisation"),
            self._record("beta", 0.99),  # high score but no matching tag
            self._record("beta", 0.99),
            self._record("beta", 0.99),
        ])
        result = engine.recommend("summarisation", "balanced")
        self.assertEqual(result.best_model, "alpha")
        self.assertTrue(result.use_case_matched)

    def test_fallback_when_no_tag_match(self):
        engine = self._engine([
            self._record("alpha", 0.9, use_case="translation"),
            self._record("alpha", 0.88, use_case="translation"),
            self._record("alpha", 0.92, use_case="translation"),
            self._record("beta", 0.75),
            self._record("beta", 0.74),
            self._record("beta", 0.76),
        ])
        result = engine.recommend("summarisation", "balanced")
        self.assertFalse(result.use_case_matched)
        # Falls back to all records, so best model is alpha (avg ~0.9 > ~0.75)
        self.assertEqual(result.best_model, "alpha")

    def test_recommend_handles_null_use_case_in_logs(self):
        engine = self._engine([
            {"model": "alpha", "score": 0.9, "use_case": None, "source": "experiment", "latency": 1.0, "cost": 0.01},
            {"model": "alpha", "score": 0.88, "use_case": None, "source": "experiment", "latency": 1.0, "cost": 0.01},
            {"model": "alpha", "score": 0.92, "use_case": None, "source": "experiment", "latency": 1.0, "cost": 0.01},
            {"model": "beta", "score": 0.75, "source": "experiment", "latency": 1.2, "cost": 0.02},
            {"model": "beta", "score": 0.74, "source": "experiment", "latency": 1.2, "cost": 0.02},
            {"model": "beta", "score": 0.76, "source": "experiment", "latency": 1.2, "cost": 0.02},
        ])

        result = engine.recommend("summarisation", "balanced")

        self.assertEqual(result.best_model, "alpha")
        self.assertFalse(result.use_case_matched)

    def test_source_filter_live_only(self):
        engine = self._engine([
            self._record("live_model", 0.9, source="live"),
            self._record("live_model", 0.88, source="live"),
            self._record("live_model", 0.92, source="live"),
            self._record("exp_model", 0.99, source="experiment"),
            self._record("exp_model", 0.99, source="experiment"),
            self._record("exp_model", 0.99, source="experiment"),
        ])
        result = engine.recommend("any", "balanced", source="live")
        self.assertEqual(result.best_model, "live_model")

    def test_source_filter_experiment_only(self):
        engine = self._engine([
            self._record("live_model", 0.99, source="live"),
            self._record("live_model", 0.99, source="live"),
            self._record("live_model", 0.99, source="live"),
            self._record("exp_model", 0.8, source="experiment"),
            self._record("exp_model", 0.82, source="experiment"),
            self._record("exp_model", 0.79, source="experiment"),
        ])
        result = engine.recommend("any", "balanced", source="experiment")
        self.assertEqual(result.best_model, "exp_model")

    def test_strategy_recommendation_uses_only_that_strategy_scores(self):
        engine = self._engine([
            {"model": "alpha", "score": 0.95, "strategy": "balanced", "source": "experiment", "latency": 1.0, "cost": 0.01},
            {"model": "alpha", "score": 0.70, "strategy": "quality", "source": "experiment", "latency": 1.0, "cost": 0.01},
            {"model": "alpha", "score": 0.72, "strategy": "quality", "source": "experiment", "latency": 1.0, "cost": 0.01},
            {"model": "alpha", "score": 0.71, "strategy": "quality", "source": "experiment", "latency": 1.0, "cost": 0.01},
            {"model": "beta", "score": 0.80, "strategy": "quality", "source": "experiment", "latency": 1.0, "cost": 0.01},
            {"model": "beta", "score": 0.82, "strategy": "quality", "source": "experiment", "latency": 1.0, "cost": 0.01},
            {"model": "beta", "score": 0.81, "strategy": "quality", "source": "experiment", "latency": 1.0, "cost": 0.01},
        ])

        result = engine.recommend("any", "quality")
        self.assertEqual(result.best_model, "beta")

    def test_min_samples_filters_models(self):
        engine = self._engine([
            self._record("alpha", 0.9),
            self._record("beta", 0.8),
            self._record("beta", 0.85),
            self._record("beta", 0.83),
        ])
        result = engine.recommend("any", "balanced", min_samples=2)
        # alpha only has 1 sample, should not appear
        self.assertEqual(result.best_model, "beta")
        self.assertEqual(len(result.alternatives), 1)

    def test_top_n_limits_alternatives(self):
        engine = self._engine([
            self._record("a", 0.9),
            self._record("b", 0.8),
            self._record("c", 0.7),
            self._record("d", 0.6),
        ])
        result = engine.recommend("any", "balanced", top_n=2)
        self.assertLessEqual(len(result.alternatives), 2)

    def test_no_data_raises_value_error(self):
        engine = self._engine([])
        with self.assertRaises(ValueError):
            engine.recommend("any", "balanced")

    def test_invalid_strategy_raises_value_error(self):
        engine = self._engine([self._record("a", 0.5)])
        with self.assertRaises(ValueError):
            engine.recommend("any", "nonexistent_strategy")

    def test_invalid_source_raises_value_error(self):
        engine = self._engine([self._record("a", 0.5)])
        with self.assertRaises(ValueError):
            engine.recommend("any", "balanced", source="invalid")

    def test_justification_includes_model_and_strategy(self):
        engine = self._engine([
            self._record("alpha", 0.9),
            self._record("alpha", 0.88),
            self._record("alpha", 0.92),
        ])
        result = engine.recommend("summarisation", "balanced")
        self.assertIn("alpha", result.justification)
        self.assertIn("balanced", result.justification)

    def test_multiple_runs_averaged(self):
        engine = self._engine([
            self._record("alpha", 0.6),
            self._record("alpha", 0.8),
            self._record("alpha", 0.7),
            self._record("beta", 0.9),
            self._record("beta", 0.88),
            self._record("beta", 0.92),
        ])
        result = engine.recommend("any", "balanced")
        # beta: avg=0.9; alpha: avg=0.7
        self.assertEqual(result.best_model, "beta")
        alpha_alt = next((a for a in result.alternatives if a.model == "alpha"), None)
        self.assertIsNotNone(alpha_alt)
        self.assertAlmostEqual(alpha_alt.score, 0.7, places=4)

    def test_sample_count_in_alternatives(self):
        engine = self._engine([
            self._record("alpha", 0.9),
            self._record("alpha", 0.85),
        ])
        result = engine.recommend("any", "balanced")
        self.assertEqual(result.alternatives[0].sample_count, 2)

    def test_p95_latency_and_consistency_in_alternatives(self):
        engine = self._engine([
            self._record("alpha", 0.9, latency=1.0),
            self._record("alpha", 0.8, latency=1.5),
            self._record("alpha", 0.6, latency=3.0),
        ])
        result = engine.recommend("any", "balanced")
        alt = result.alternatives[0]
        self.assertGreaterEqual(alt.p95_latency, 1.5)
        self.assertAlmostEqual(alt.consistency_above_threshold, 2 / 3, places=4)

    def test_confidence_rewards_more_samples(self):
        engine = self._engine([self._record("alpha", 0.8)])

        low_sample_confidence = engine._confidence(
            sample_count=5,
            score_stddev=0.05,
            consistency_above_threshold=0.8,
            score_delta_from_best=0.0,
        )
        high_sample_confidence = engine._confidence(
            sample_count=50,
            score_stddev=0.05,
            consistency_above_threshold=0.8,
            score_delta_from_best=0.0,
        )

        self.assertGreater(high_sample_confidence, low_sample_confidence)

    def test_confidence_is_high_for_strong_evidence(self):
        engine = self._engine([self._record("alpha", 0.8)])

        confidence = engine._confidence(
            sample_count=60,
            score_stddev=0.1237,
            consistency_above_threshold=0.98,
            score_delta_from_best=0.0,
        )

        self.assertGreaterEqual(confidence, 0.8)

    def test_confidence_guardrails_cap_small_sample_recommendations(self):
        engine = self._engine([
            self._record("alpha", 0.95),
            self._record("alpha", 0.92),
            self._record("beta", 0.80),
            self._record("beta", 0.79),
        ])

        result = engine.recommend("any", "balanced")

        self.assertEqual(result.confidence_label, "INSUFFICIENT EVIDENCE")
        self.assertLessEqual(result.confidence, 0.34)
        self.assertTrue(result.confidence_reasons)

    def test_confidence_guardrails_cap_narrow_wins(self):
        engine = self._engine([
            self._record("alpha", 0.82),
            self._record("alpha", 0.81),
            self._record("alpha", 0.80),
            self._record("alpha", 0.82),
            self._record("alpha", 0.81),
            self._record("beta", 0.80),
            self._record("beta", 0.80),
            self._record("beta", 0.79),
            self._record("beta", 0.81),
            self._record("beta", 0.80),
        ])

        result = engine.recommend("any", "balanced")

        self.assertLessEqual(result.confidence, 0.49)
        self.assertIn("second-best model", " ".join(result.confidence_reasons))

    def test_validity_gate_marks_invalid_for_too_few_samples(self):
        engine = self._engine([
            self._record("alpha", 0.91),
            self._record("alpha", 0.90),
            self._record("beta", 0.80),
            self._record("beta", 0.79),
        ])

        result = engine.recommend("any", "balanced")

        self.assertFalse(result.is_valid)
        self.assertEqual(result.validity_status, "INVALID")
        self.assertIn("INSUFFICIENT_DATA", result.failure_modes)

    def test_validity_gate_marks_warning_for_narrow_margin(self):
        engine = self._engine([
            self._record("alpha", 0.81),
            self._record("alpha", 0.82),
            self._record("alpha", 0.83),
            self._record("alpha", 0.81),
            self._record("alpha", 0.82),
            self._record("alpha", 0.83),
            self._record("alpha", 0.81),
            self._record("alpha", 0.82),
            self._record("alpha", 0.83),
            self._record("alpha", 0.81),
            self._record("beta", 0.79),
            self._record("beta", 0.80),
            self._record("beta", 0.81),
            self._record("beta", 0.79),
            self._record("beta", 0.80),
            self._record("beta", 0.81),
            self._record("beta", 0.79),
            self._record("beta", 0.80),
            self._record("beta", 0.81),
            self._record("beta", 0.80),
        ])

        result = engine.recommend("any", "balanced")

        self.assertEqual(result.validity_status, "WARNING")
        self.assertFalse(result.is_valid)
        self.assertIn("LOW_SEPARATION", result.failure_modes)

    def test_validity_gate_marks_valid_when_evidence_is_strong(self):
        engine = self._engine([
            self._record("alpha", 0.90, use_case="any"),
            self._record("alpha", 0.91, use_case="any"),
            self._record("alpha", 0.89, use_case="any"),
            self._record("alpha", 0.92, use_case="any"),
            self._record("alpha", 0.90, use_case="any"),
            self._record("alpha", 0.91, use_case="any"),
            self._record("alpha", 0.90, use_case="any"),
            self._record("alpha", 0.92, use_case="any"),
            self._record("alpha", 0.91, use_case="any"),
            self._record("alpha", 0.90, use_case="any"),
            self._record("beta", 0.72, use_case="any"),
            self._record("beta", 0.71, use_case="any"),
            self._record("beta", 0.70, use_case="any"),
            self._record("beta", 0.73, use_case="any"),
            self._record("beta", 0.72, use_case="any"),
            self._record("beta", 0.71, use_case="any"),
            self._record("beta", 0.70, use_case="any"),
            self._record("beta", 0.72, use_case="any"),
            self._record("beta", 0.71, use_case="any"),
            self._record("beta", 0.70, use_case="any"),
        ])

        result = engine.recommend("any", "balanced")

        self.assertEqual(result.validity_status, "VALID")
        self.assertTrue(result.is_valid)
        self.assertEqual(result.failure_modes, [])

    def test_no_recommendation_when_all_models_weak(self):
        engine = self._engine([
            self._record("alpha", 0.55, use_case="summarisation"),
            self._record("alpha", 0.52, use_case="summarisation"),
            self._record("alpha", 0.58, use_case="summarisation"),
            self._record("beta", 0.49, use_case="summarisation"),
            self._record("beta", 0.51, use_case="summarisation"),
            self._record("beta", 0.50, use_case="summarisation"),
        ])

        result = engine.recommend("summarisation", "balanced")

        self.assertFalse(result.recommendation_available)
        self.assertEqual(result.best_model, "")
        self.assertEqual(result.validity_status, "INVALID")
        self.assertIn("ALL_MODELS_WEAK", result.failure_modes)
        self.assertTrue(result.no_recommendation_reason)

    def test_quality_strategy_uses_stricter_weak_threshold(self):
        engine = self._engine([
            self._record("alpha", 0.75, use_case="summarisation"),
            self._record("alpha", 0.76, use_case="summarisation"),
            self._record("alpha", 0.74, use_case="summarisation"),
            self._record("beta", 0.72, use_case="summarisation"),
            self._record("beta", 0.73, use_case="summarisation"),
            self._record("beta", 0.71, use_case="summarisation"),
        ])

        result = engine.recommend("summarisation", "quality")

        self.assertFalse(result.recommendation_available)
        self.assertIn("ALL_MODELS_WEAK", result.failure_modes)

    def test_cost_aware_strategy_uses_looser_weak_threshold(self):
        # Scores around 0.72 are above cost_aware weak threshold (0.58) but below
        # quality threshold (0.78), demonstrating strategy-specific leniency.
        engine = self._engine([
            self._record("alpha", 0.74, use_case="summarisation"),
            self._record("alpha", 0.75, use_case="summarisation"),
            self._record("alpha", 0.73, use_case="summarisation"),
            self._record("beta", 0.71, use_case="summarisation"),
            self._record("beta", 0.72, use_case="summarisation"),
            self._record("beta", 0.70, use_case="summarisation"),
        ])

        result = engine.recommend("summarisation", "cost_aware")

        self.assertTrue(result.recommendation_available)
        self.assertNotIn("ALL_MODELS_WEAK", result.failure_modes)

    def test_evaluation_status_insufficient_data_when_sample_volume_low(self):
        engine = self._engine([self._record("alpha", 0.8)])

        status = engine._derive_evaluation_status(
            validity_status="WARNING",
            sample_count=4,
            score_stddev=0.01,
            margin_to_runner_up=0.2,
            best_score=0.9,
            consistency_above_threshold=0.95,
        )

        self.assertEqual(status, "INSUFFICIENT_DATA")

    def test_evaluation_status_noisy_for_extreme_variance(self):
        engine = self._engine([self._record("alpha", 0.8)])

        status = engine._derive_evaluation_status(
            validity_status="WARNING",
            sample_count=12,
            score_stddev=0.25,
            margin_to_runner_up=0.12,
            best_score=0.88,
            consistency_above_threshold=0.9,
        )

        self.assertEqual(status, "NOISY")

    def test_evaluation_status_unstable_for_elevated_variance(self):
        engine = self._engine([self._record("alpha", 0.8)])

        status = engine._derive_evaluation_status(
            validity_status="WARNING",
            sample_count=12,
            score_stddev=0.16,
            margin_to_runner_up=0.12,
            best_score=0.88,
            consistency_above_threshold=0.9,
        )

        self.assertEqual(status, "UNSTABLE")

    def test_evaluation_status_weak_signal_for_low_margin(self):
        engine = self._engine([self._record("alpha", 0.8)])

        status = engine._derive_evaluation_status(
            validity_status="WARNING",
            sample_count=12,
            score_stddev=0.05,
            margin_to_runner_up=0.02,
            best_score=0.88,
            consistency_above_threshold=0.9,
        )

        self.assertEqual(status, "WEAK_SIGNAL")

    def test_system_state_transition_matrix(self):
        engine = self._engine([self._record("alpha", 0.8)])

        matrix = [
            ("RECOMMENDED", "DECIDING", "RECOMMENDED"),
            ("CONSTRAINED_RECOMMENDATION", "CONSTRAINED_DECISION", "CONSTRAINED"),
            ("ABSTAIN", "ABSTAINING", "NONE"),
            ("INVALID", "INVALID_DATA", "NONE"),
            ("UNKNOWN_STATE", "DECIDING", "NONE"),
        ]

        for decision_state, expected_system_state, expected_decision_result in matrix:
            with self.subTest(decision_state=decision_state):
                system_state, decision_result = engine._derive_system_decision_outputs(decision_state)
                self.assertEqual(system_state, expected_system_state)
                self.assertEqual(decision_result, expected_decision_result)


if __name__ == "__main__":
    unittest.main()
