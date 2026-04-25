"""Recommendation engine.

Reads experiment and/or live logs from ``experiments/logs.jsonl``, aggregates
per-model statistics (average score per scoring strategy, latency, cost), and
ranks models for a caller-supplied use_case / strategy combination.

A ``use_case`` tag is stored in log entries when present (written by the runner
or compare endpoint via the ``use_case`` field).  If no entries carry the
requested tag the engine falls back to all entries in the chosen source scope.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from math import ceil, log1p
from statistics import pstdev
from typing import Dict, List, Optional

from backend.scoring.registry import SCORERS

_logger = logging.getLogger(__name__)

LOG_PATH = Path("experiments/logs.jsonl")
RECOMMENDATION_LOG_PATH = Path("recommendations/recommendations.jsonl")

_VALID_SOURCES = {"live", "experiment", "all"}
_DEFAULT_WEAK_SCORE_THRESHOLD = 0.70
_VALIDITY_THRESHOLD = 0.55
_DECISION_THRESHOLD = 0.70
_DISPLAY_THRESHOLD = 0.40
_WEAK_SCORE_THRESHOLDS = {
    "quality": 0.78,
    "rag": 0.74,
    "balanced": 0.70,
    "latency_aware": 0.64,
    "cost_aware": 0.58,
}


@dataclass
class ModelStats:
    model: str
    scores: Dict[str, List[float]] = field(default_factory=dict)
    latencies: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    total_tokens: List[float] = field(default_factory=list)
    cost_per_1k_tokens: List[float] = field(default_factory=list)
    quality_per_1k_tokens: List[float] = field(default_factory=list)

    def avg_score(self, strategy: str) -> float:
        vals = self.scores.get(strategy, [])
        return sum(vals) / len(vals) if vals else 0.0

    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def avg_cost(self) -> float:
        return sum(self.costs) / len(self.costs) if self.costs else 0.0

    def avg_total_tokens(self) -> float:
        return sum(self.total_tokens) / len(self.total_tokens) if self.total_tokens else 0.0

    def avg_cost_per_1k_tokens(self) -> float:
        return sum(self.cost_per_1k_tokens) / len(self.cost_per_1k_tokens) if self.cost_per_1k_tokens else 0.0

    def avg_quality_per_1k_tokens(self) -> float:
        return sum(self.quality_per_1k_tokens) / len(self.quality_per_1k_tokens) if self.quality_per_1k_tokens else 0.0

    def sample_count(self, strategy: str) -> int:
        return len(self.scores.get(strategy, []))

    def score_stddev(self, strategy: str) -> float:
        vals = self.scores.get(strategy, [])
        if len(vals) <= 1:
            return 0.0
        return float(pstdev(vals))

    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        idx = max(0, min(len(ordered) - 1, ceil(0.95 * len(ordered)) - 1))
        return float(ordered[idx])

    def consistency_above_threshold(self, strategy: str, threshold: float = 0.7) -> float:
        vals = self.scores.get(strategy, [])
        if not vals:
            return 0.0
        passing = sum(1 for score in vals if score >= threshold)
        return passing / len(vals)


@dataclass
class ModelRecommendation:
    model: str
    score: float
    sample_count: int
    avg_latency: float
    avg_cost: float
    score_stddev: float = 0.0
    score_delta_from_best: float = 0.0
    confidence: float = 0.0
    p95_latency: float = 0.0
    consistency_above_threshold: float = 0.0
    avg_total_tokens: float = 0.0
    avg_cost_per_1k_tokens: float = 0.0
    avg_quality_per_1k_tokens: float = 0.0


@dataclass
class RecommendationResult:
    best_model: str
    best_score: float
    strategy: str
    use_case: str
    use_case_matched: bool
    system_state: str
    decision_result: str
    decision_state: str
    evaluation_status: str
    decision_reliability: float
    validity_threshold: float
    decision_threshold: float
    display_threshold: float
    display_eligible: bool
    score_summary: Dict[str, float]
    score_vector: Dict[str, float]
    recommendation_available: bool
    no_recommendation_reason: Optional[str]
    gate_status: str
    gate_threshold: float
    gate_triggers: List[str]
    score_margin: float
    score_variance: float
    validity_status: str
    is_valid: bool
    validity_reasons: List[str]
    failure_modes: List[str]
    confidence: float
    confidence_label: str
    confidence_reasons: List[str]
    alternatives: List[ModelRecommendation]
    justification: str


def _seed_strategy_scores(record: dict, target_strategy: Optional[str] = None) -> Dict[str, float]:
    """Derive per-strategy scores from a log record.

    If the record already has ``scores_by_strategy``, use those directly.
    Otherwise derive the strategy from the record's ``strategy`` field. As a
    fallback for older logs that only have a single score, bind the score to
    the caller-provided ``target_strategy`` (or ``balanced`` as last resort).
    """
    if isinstance(record.get("scores_by_strategy"), dict):
        return {
            k: float(v)
            for k, v in record["scores_by_strategy"].items()
            if k in SCORERS and isinstance(v, (int, float))
        }
    raw_score = record.get("score")
    if isinstance(raw_score, (int, float)):
        record_strategy = record.get("strategy")
        if isinstance(record_strategy, str) and record_strategy in SCORERS:
            return {record_strategy: float(raw_score)}
        if isinstance(target_strategy, str) and target_strategy in SCORERS:
            return {target_strategy: float(raw_score)}
        return {"balanced": float(raw_score)}
    return {}


class RecommendationEngine:
    def __init__(self, log_path: Path = LOG_PATH) -> None:
        self._log_path = log_path

    def _weak_threshold_for_strategy(self, strategy: str) -> float:
        return float(_WEAK_SCORE_THRESHOLDS.get(strategy, _DEFAULT_WEAK_SCORE_THRESHOLD))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        use_case: str,
        strategy: str,
        top_n: int = 3,
        min_samples: int = 1,
        source: str = "all",
    ) -> RecommendationResult:
        if strategy not in SCORERS:
            raise ValueError(f"Unknown scoring strategy: {strategy!r}. Valid: {sorted(SCORERS)}")
        if source not in _VALID_SOURCES:
            raise ValueError(f"Unknown source: {source!r}. Valid: {sorted(_VALID_SOURCES)}")

        records = self._load_records(source)

        # Try to match by use_case tag; fall back to all records.
        requested_use_case = str(use_case or "").strip().lower()
        tagged = [
            r
            for r in records
            if str(r.get("use_case") or "").strip().lower() == requested_use_case
        ]
        use_case_matched = bool(tagged)
        working_records = tagged if use_case_matched else records

        stats = self._aggregate(working_records, strategy, min_samples)
        if not stats:
            raise ValueError(
                f"No qualifying model data found for strategy={strategy!r}, "
                f"source={source!r}, min_samples={min_samples}."
            )

        ranked = sorted(stats, key=lambda s: -s.avg_score(strategy))
        top = ranked[:top_n]
        weak_threshold = self._weak_threshold_for_strategy(strategy)
        all_models_weak = all(s.avg_score(strategy) < weak_threshold for s in ranked)

        best = top[0]
        best_score = best.avg_score(strategy)
        runner_up_score = top[1].avg_score(strategy) if len(top) > 1 else None
        margin_to_runner_up = (best_score - runner_up_score) if runner_up_score is not None else 0.0
        best_score_stddev = best.score_stddev(strategy)
        best_consistency = best.consistency_above_threshold(strategy)
        alternatives = []
        for s in top:
            avg_score = s.avg_score(strategy)
            sample_count = s.sample_count(strategy)
            score_stddev = s.score_stddev(strategy)
            score_delta_from_best = avg_score - best_score
            consistency = s.consistency_above_threshold(strategy)

            alternatives.append(
                ModelRecommendation(
                    model=s.model,
                    score=round(avg_score, 4),
                    sample_count=sample_count,
                    avg_latency=round(s.avg_latency(), 4),
                    avg_cost=round(s.avg_cost(), 8),
                    score_stddev=round(score_stddev, 4),
                    score_delta_from_best=round(score_delta_from_best, 4),
                    confidence=round(
                        self._confidence(
                            sample_count=sample_count,
                            score_stddev=score_stddev,
                            consistency_above_threshold=consistency,
                            score_delta_from_best=score_delta_from_best,
                        ),
                        4,
                    ),
                    p95_latency=round(s.p95_latency(), 4),
                    consistency_above_threshold=round(consistency, 4),
                    avg_total_tokens=round(s.avg_total_tokens(), 2),
                    avg_cost_per_1k_tokens=round(s.avg_cost_per_1k_tokens(), 6),
                    avg_quality_per_1k_tokens=round(s.avg_quality_per_1k_tokens(), 4),
                )
            )

        best_confidence = alternatives[0].confidence
        confidence, confidence_label, confidence_reasons = self._apply_confidence_guardrails(
            raw_confidence=best_confidence,
            best_score=best_score,
            sample_count=best.sample_count(strategy),
            score_stddev=best_score_stddev,
            consistency_above_threshold=best_consistency,
            margin_to_runner_up=margin_to_runner_up,
            use_case_matched=use_case_matched,
        )
        alternatives[0].confidence = round(confidence, 4)

        validity_status, validity_reasons = self._evaluate_validity_gate(
            best_score=best_score,
            sample_count=best.sample_count(strategy),
            score_stddev=best_score_stddev,
            consistency_above_threshold=best_consistency,
            margin_to_runner_up=margin_to_runner_up,
            use_case_matched=use_case_matched,
        )
        failure_modes = self._classify_failure_modes(
            strategy=strategy,
            best_score=best_score,
            sample_count=best.sample_count(strategy),
            score_stddev=best_score_stddev,
            consistency_above_threshold=best_consistency,
            margin_to_runner_up=margin_to_runner_up,
            use_case_matched=use_case_matched,
        )
        recommendation_available = not all_models_weak
        no_recommendation_reason: Optional[str] = None
        gate_triggers: List[str] = []
        if all_models_weak:
            gate_triggers.append("ALL_MODELS_WEAK")
            no_recommendation_reason = (
                f"All qualifying models are weak for strategy '{strategy}' "
                f"(best score {best_score:.4f} < {weak_threshold:.2f})."
            )
            if "ALL_MODELS_WEAK" not in failure_modes:
                failure_modes.append("ALL_MODELS_WEAK")
            validity_status = "INVALID"
            confidence = min(confidence, 0.39)
            confidence_label = "LOW"
            confidence_reasons.append(
                f"No model met the minimum strength threshold ({weak_threshold:.2f})."
            )

        if validity_status == "INVALID":
            decision_state = "INVALID"
        elif not recommendation_available:
            decision_state = "ABSTAIN"
        elif validity_status != "VALID":
            decision_state = "CONSTRAINED_RECOMMENDATION"
        else:
            decision_state = "RECOMMENDED"

        # Derive recommendation_available consistently from decision_state so it
        # is never True for INVALID or ABSTAIN states, regardless of which gate
        # triggered the INVALID condition (e.g. <3 samples, below validity
        # threshold, or ALL_MODELS_WEAK).
        recommendation_available = decision_state in {
            "RECOMMENDED",
            "CONSTRAINED_RECOMMENDATION",
        }

        system_state, decision_result = self._derive_system_decision_outputs(decision_state)

        evaluation_status = self._derive_evaluation_status(
            validity_status=validity_status,
            sample_count=best.sample_count(strategy),
            score_stddev=best_score_stddev,
            margin_to_runner_up=margin_to_runner_up,
            best_score=best_score,
            consistency_above_threshold=best_consistency,
        )

        decision_reliability = round(confidence, 4)
        score_summary = {
            "best_score": round(best_score, 4),
            "margin": round(margin_to_runner_up, 4),
            "variance": round(best_score_stddev, 4),
        }
        score_vector = {
            "quality_score": round(best_score, 4),
            "latency": round(best.avg_latency(), 4),
            "cost": round(best.avg_cost(), 8),
            "variance": round(best_score_stddev, 4),
            "sample_size": float(best.sample_count(strategy)),
        }

        justification = self._build_justification(
            best,
            strategy,
            use_case,
            use_case_matched,
            alternatives,
            recommendation_available=recommendation_available,
            no_recommendation_reason=no_recommendation_reason,
        )

        result = RecommendationResult(
            best_model=best.model if recommendation_available else "",
            best_score=round(best.avg_score(strategy), 4),
            strategy=strategy,
            use_case=use_case,
            use_case_matched=use_case_matched,
            system_state=system_state,
            decision_result=decision_result,
            decision_state=decision_state,
            evaluation_status=evaluation_status,
            decision_reliability=decision_reliability,
            validity_threshold=_VALIDITY_THRESHOLD,
            decision_threshold=_DECISION_THRESHOLD,
            display_threshold=_DISPLAY_THRESHOLD,
            display_eligible=(decision_reliability >= _DISPLAY_THRESHOLD),
            score_summary=score_summary,
            score_vector=score_vector,
            recommendation_available=recommendation_available,
            no_recommendation_reason=no_recommendation_reason,
            gate_status="ABSTAIN" if all_models_weak else "PASS",
            gate_threshold=round(weak_threshold, 4),
            gate_triggers=gate_triggers,
            score_margin=round(margin_to_runner_up, 4),
            score_variance=round(best_score_stddev, 4),
            validity_status=validity_status,
            is_valid=(validity_status == "VALID" and recommendation_available),
            validity_reasons=validity_reasons,
            failure_modes=failure_modes,
            confidence=round(confidence, 4),
            confidence_label=confidence_label,
            confidence_reasons=confidence_reasons,
            alternatives=alternatives,
            justification=justification,
        )

        self._persist(result, source)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_records(self, source: str) -> List[dict]:
        if not self._log_path.exists():
            return []
        records: List[dict] = []
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not record.get("model"):
                    continue
                rec_source = str(record.get("source", "experiment")).lower()
                if source == "all" or rec_source == source:
                    records.append(record)
        return records

    def _aggregate(
        self,
        records: List[dict],
        strategy: str,
        min_samples: int,
    ) -> List[ModelStats]:
        stats_map: Dict[str, ModelStats] = {}
        for record in records:
            model = record.get("model")
            if not model:
                continue
            if model not in stats_map:
                stats_map[model] = ModelStats(model=model)
            s = stats_map[model]

            strategy_scores = _seed_strategy_scores(record, strategy)
            for strat, score_val in strategy_scores.items():
                s.scores.setdefault(strat, []).append(score_val)

            latency = record.get("latency")
            if isinstance(latency, (int, float)):
                s.latencies.append(float(latency))

            cost = record.get("cost")
            if isinstance(cost, (int, float)):
                s.costs.append(float(cost))

            total_tokens = record.get("total_tokens")
            if isinstance(total_tokens, (int, float)) and float(total_tokens) > 0:
                s.total_tokens.append(float(total_tokens))

            record_cost_per_1k = record.get("cost_per_1k_tokens")
            if isinstance(record_cost_per_1k, (int, float)):
                s.cost_per_1k_tokens.append(float(record_cost_per_1k))

            metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
            metric_cost_per_1k = metrics.get("cost_per_1k_tokens")
            if isinstance(metric_cost_per_1k, (int, float)) and not isinstance(record_cost_per_1k, (int, float)):
                s.cost_per_1k_tokens.append(float(metric_cost_per_1k))

            metric_quality_per_1k = metrics.get("quality_per_1k_tokens")
            if isinstance(metric_quality_per_1k, (int, float)):
                s.quality_per_1k_tokens.append(float(metric_quality_per_1k))

            metric_total_tokens = metrics.get("total_token_count")
            if isinstance(metric_total_tokens, (int, float)) and not isinstance(total_tokens, (int, float)):
                s.total_tokens.append(float(metric_total_tokens))

        return [s for s in stats_map.values() if s.sample_count(strategy) >= min_samples]

    def _build_justification(
        self,
        best: ModelStats,
        strategy: str,
        use_case: str,
        use_case_matched: bool,
        alternatives: List[ModelRecommendation],
        recommendation_available: bool,
        no_recommendation_reason: Optional[str],
    ) -> str:
        scope = f"entries tagged use_case='{use_case}'" if use_case_matched else "all available entries (no use_case tag match found)"
        parts = [
            (
                f"Recommended '{best.model}' for strategy '{strategy}' based on {scope}."
                if recommendation_available
                else f"No recommendation available for strategy '{strategy}' based on {scope}."
            ),
            f"Average score: {best.avg_score(strategy):.4f} over {best.sample_count(strategy)} sample(s).",
            f"Score variability (std dev): {best.score_stddev(strategy):.4f}; confidence={alternatives[0].confidence:.2f} ({self._confidence_label(alternatives[0].confidence, best.sample_count(strategy))}).",
            f"Average latency: {best.avg_latency():.3f}s (p95={best.p95_latency():.3f}s), average cost: {best.avg_cost():.6f}.",
            f"Consistency above 0.7 score: {best.consistency_above_threshold(strategy) * 100:.1f}%.",
        ]
        if no_recommendation_reason:
            parts.append(no_recommendation_reason)
        if len(alternatives) > 1:
            alt_names = ", ".join(
                f"'{a.model}' ({a.score:.4f}, delta={a.score_delta_from_best:+.4f}, conf={a.confidence:.2f})"
                for a in alternatives[1:]
            )
            parts.append(f"Alternatives considered: {alt_names}.")
        return " ".join(parts)

    def _confidence(
        self,
        sample_count: int,
        score_stddev: float,
        consistency_above_threshold: float,
        score_delta_from_best: float,
    ) -> float:
        # Confidence should reward more evidence and stable quality, while only
        # softly penalizing moderate variance. Models below the best score also
        # lose confidence because the recommendation case for them is weaker.
        sample_factor = min(1.0, max(0.0, log1p(float(sample_count)) / log1p(50.0)))
        stability_factor = 1.0 / (1.0 + max(0.0, float(score_stddev)) * 2.5)
        consistency_factor = max(0.0, min(1.0, float(consistency_above_threshold)))
        gap_penalty = min(1.0, max(0.0, abs(float(score_delta_from_best)) / 0.3))

        weighted = (
            0.35 * sample_factor
            + 0.25 * stability_factor
            + 0.40 * consistency_factor
        )

        adjusted = weighted * (1.0 - (0.35 * gap_penalty))
        return max(0.0, min(1.0, adjusted))

    def _apply_confidence_guardrails(
        self,
        raw_confidence: float,
        best_score: float,
        sample_count: int,
        score_stddev: float,
        consistency_above_threshold: float,
        margin_to_runner_up: Optional[float],
        use_case_matched: bool,
    ) -> tuple[float, str, List[str]]:
        guarded = max(0.0, min(1.0, float(raw_confidence)))
        reasons: List[str] = []

        if sample_count < 3:
            guarded = min(guarded, 0.34)
            reasons.append("Fewer than 3 samples are available, so this is not yet strong evidence.")
        elif sample_count < 5:
            guarded = min(guarded, 0.54)
            reasons.append("Fewer than 5 samples are available, so confidence is capped.")
        elif sample_count < 10:
            guarded = min(guarded, 0.74)
            reasons.append("Fewer than 10 samples are available, so high confidence is withheld.")

        if margin_to_runner_up is not None:
            if margin_to_runner_up < 0.03:
                guarded = min(guarded, 0.49)
                reasons.append("The lead over the second-best model is very small.")
            elif margin_to_runner_up < 0.08:
                guarded = min(guarded, 0.69)
                reasons.append("The lead over the second-best model is still modest.")

        if best_score < 0.70:
            guarded = min(guarded, 0.59)
            reasons.append("The recommended model is below the default quality threshold of 0.70.")

        if consistency_above_threshold < 0.70:
            guarded = min(guarded, 0.59)
            reasons.append("Too few runs clear the 0.70 score threshold consistently.")

        if score_stddev > 0.18:
            guarded = min(guarded, 0.54)
            reasons.append("Score variability is high, which weakens the reliability of the recommendation.")

        if not use_case_matched:
            guarded = min(guarded, 0.64)
            reasons.append("The recommendation fell back to general logs because no exact use-case match was found.")

        return round(guarded, 4), self._confidence_label(guarded, sample_count), reasons

    def _confidence_label(self, confidence: float, sample_count: int) -> str:
        if sample_count < 3:
            return "INSUFFICIENT EVIDENCE"
        if confidence >= 0.8:
            return "HIGH"
        if confidence >= 0.55:
            return "MEDIUM"
        return "LOW"

    def _evaluate_validity_gate(
        self,
        best_score: float,
        sample_count: int,
        score_stddev: float,
        consistency_above_threshold: float,
        margin_to_runner_up: float,
        use_case_matched: bool,
    ) -> tuple[str, List[str]]:
        reasons: List[str] = []

        if best_score < -1.0 or best_score > 1.0:
            reasons.append("Best score is outside the expected [-1.0, 1.0] range.")
            return "INVALID", reasons
        if sample_count < 3:
            reasons.append("Less than 3 samples are available for the recommended model.")
            return "INVALID", reasons
        if best_score < _VALIDITY_THRESHOLD:
            reasons.append(f"Best score is below the minimum validity threshold ({_VALIDITY_THRESHOLD:.2f}).")
            return "INVALID", reasons
        if consistency_above_threshold < 0.55:
            reasons.append("Consistency is too low (<55% runs above score 0.7).")
            return "INVALID", reasons

        if sample_count < 10:
            reasons.append("Evidence volume is moderate (<10 samples).")
        if margin_to_runner_up < 0.08:
            reasons.append("Score margin to the second-best model is modest.")
        if score_stddev > 0.15:
            reasons.append("Score variability is elevated.")
        if not use_case_matched:
            reasons.append("Recommendation used fallback logs with no exact use-case match.")

        if reasons:
            return "WARNING", reasons

        return "VALID", ["All validity checks passed."]

    def _derive_evaluation_status(
        self,
        validity_status: str,
        sample_count: int,
        score_stddev: float,
        margin_to_runner_up: float,
        best_score: float,
        consistency_above_threshold: float,
    ) -> str:
        # Semantic mapping:
        # - INSUFFICIENT_DATA: not enough evidence volume
        # - NOISY: extreme variance
        # - UNSTABLE: elevated variance (not extreme)
        # - WEAK_SIGNAL: evidence exists but quality/separation/consistency is weak
        # - VALID: reliable and decision-worthy evidence
        if validity_status == "INVALID":
            if sample_count < 3:
                return "INSUFFICIENT_DATA"
            if score_stddev >= 0.20:
                return "NOISY"
            return "WEAK_SIGNAL"

        if sample_count < 5:
            return "INSUFFICIENT_DATA"
        if score_stddev >= 0.20:
            return "NOISY"
        if score_stddev >= 0.15:
            return "UNSTABLE"
        if (
            best_score < _DECISION_THRESHOLD
            or margin_to_runner_up < 0.03
            or consistency_above_threshold < 0.70
        ):
            return "WEAK_SIGNAL"
        return "VALID"

    def _derive_system_decision_outputs(self, decision_state: str) -> tuple[str, str]:
        system_state = {
            "RECOMMENDED": "DECIDING",
            "CONSTRAINED_RECOMMENDATION": "CONSTRAINED_DECISION",
            "ABSTAIN": "ABSTAINING",
            "INVALID": "INVALID_DATA",
        }.get(decision_state, "DECIDING")

        decision_result = {
            "RECOMMENDED": "RECOMMENDED",
            "CONSTRAINED_RECOMMENDATION": "CONSTRAINED",
            "ABSTAIN": "NONE",
            "INVALID": "NONE",
        }.get(decision_state, "NONE")

        return system_state, decision_result

    def _classify_failure_modes(
        self,
        strategy: str,
        best_score: float,
        sample_count: int,
        score_stddev: float,
        consistency_above_threshold: float,
        margin_to_runner_up: float,
        use_case_matched: bool,
    ) -> List[str]:
        modes: List[str] = []
        if best_score < -1.0 or best_score > 1.0:
            modes.append("SCORE_SCALE_DRIFT")
        if sample_count < 5:
            modes.append("INSUFFICIENT_DATA")
        if margin_to_runner_up < 0.03:
            modes.append("LOW_SEPARATION")
        if score_stddev > 0.18:
            modes.append("HIGH_VARIANCE")
        if best_score < 0.70:
            modes.append("LOW_QUALITY")
        weak_threshold = self._weak_threshold_for_strategy(strategy)
        if best_score < weak_threshold:
            modes.append("ALL_MODELS_WEAK")
        if consistency_above_threshold < 0.70:
            modes.append("LOW_CONSISTENCY")
        if not use_case_matched:
            modes.append("USE_CASE_MISMATCH")
        return modes

    def _persist(self, result: RecommendationResult, source: str) -> None:
        if os.getenv("RECOMMENDATION_AUDIT_LOG", "1").lower() in ("0", "false", "no"):
            return
        RECOMMENDATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "use_case": result.use_case,
            "strategy": result.strategy,
            "source_scope": source,
            "use_case_matched": result.use_case_matched,
            "system_state": result.system_state,
            "decision_result": result.decision_result,
            "decision_state": result.decision_state,
            "evaluation_status": result.evaluation_status,
            "decision_reliability": result.decision_reliability,
            "validity_threshold": result.validity_threshold,
            "decision_threshold": result.decision_threshold,
            "display_threshold": result.display_threshold,
            "display_eligible": result.display_eligible,
            "score_summary": result.score_summary,
            "score_vector": result.score_vector,
            "recommendation_available": result.recommendation_available,
            "no_recommendation_reason": result.no_recommendation_reason,
            "gate_status": result.gate_status,
            "gate_threshold": result.gate_threshold,
            "gate_triggers": result.gate_triggers,
            "best_model": result.best_model,
            "best_score": result.best_score,
            "score_margin": result.score_margin,
            "score_variance": result.score_variance,
            "validity_status": result.validity_status,
            "is_valid": result.is_valid,
            "validity_reasons": result.validity_reasons,
            "failure_modes": result.failure_modes,
            "confidence": result.confidence,
            "confidence_label": result.confidence_label,
            "confidence_reasons": result.confidence_reasons,
            "alternatives": [
                {
                    "model": a.model,
                    "score": a.score,
                    "sample_count": a.sample_count,
                    "avg_latency": a.avg_latency,
                    "avg_cost": a.avg_cost,
                    "score_stddev": a.score_stddev,
                    "score_delta_from_best": a.score_delta_from_best,
                    "confidence": a.confidence,
                    "p95_latency": a.p95_latency,
                    "consistency_above_threshold": a.consistency_above_threshold,
                    "avg_total_tokens": a.avg_total_tokens,
                    "avg_cost_per_1k_tokens": a.avg_cost_per_1k_tokens,
                    "avg_quality_per_1k_tokens": a.avg_quality_per_1k_tokens,
                }
                for a in result.alternatives
            ],
            "justification": result.justification,
        }
        try:
            with open(RECOMMENDATION_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            _logger.warning("Recommendation audit log write failed: %s", exc)
