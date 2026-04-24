from fastapi import APIRouter, HTTPException

from backend.api.schemas import (
    AlternativeModel,
    Decision,
    DecisionReliability,
    EvaluationMetrics,
    EvaluationResponse,
    FailureAnalysis,
    ModelAlternative,
    RecommendationRequest,
    RecommendationResponse,
    ScoreVector,
)
from backend.recommender.engine import RecommendationEngine

router = APIRouter()
_engine = RecommendationEngine()


def _map_decision_state(state: str) -> str:
    return {
        "RECOMMENDED": "RECOMMENDED",
        "CONSTRAINED_RECOMMENDATION": "CONSTRAINED",
        "ABSTAIN": "ABSTAIN",
        "INVALID": "INVALID",
    }.get(state, "ABSTAIN")


def _map_system_health(decision_state: str) -> str:
    return {
        "RECOMMENDED": "OK",
        "CONSTRAINED_RECOMMENDATION": "DEGRADED",
        "ABSTAIN": "DEGRADED",
        "INVALID": "FAIL",
    }.get(decision_state, "DEGRADED")


def _map_evaluation_status(status: str) -> str:
    if status in {"VALID", "INVALID", "INSUFFICIENT_DATA", "NOISY", "UNSTABLE", "WEAK_SIGNAL"}:
        return status
    return "WEAK_SIGNAL"


def _map_reliability_label(label: str) -> str:
    if label in {"HIGH", "MEDIUM", "LOW", "INSUFFICIENT EVIDENCE"}:
        return label
    return "LOW"


def _map_failure_mode(mode: str) -> str:
    if mode in {
        "LOW_QUALITY",
        "LOW_SEPARATION",
        "INSUFFICIENT_DATA",
        "HIGH_VARIANCE",
        "COST_DOMINATED",
        "LATENCY_DOMINATED",
        "ALL_MODELS_WEAK",
        "LOW_CONSISTENCY",
        "USE_CASE_MISMATCH",
        "SCORE_SCALE_DRIFT",
    }:
        return mode
    return {
        "low_quality": "LOW_QUALITY",
        "all_models_weak": "ALL_MODELS_WEAK",
        "weak_separation": "LOW_SEPARATION",
        "low_consistency": "LOW_CONSISTENCY",
        "use_case_mismatch": "USE_CASE_MISMATCH",
        "insufficient_data": "INSUFFICIENT_DATA",
        "high_variance": "HIGH_VARIANCE",
        "score_scale_drift": "SCORE_SCALE_DRIFT",
        "cost_dominated": "COST_DOMINATED",
        "latency_dominated": "LATENCY_DOMINATED",
    }.get(mode, "LOW_QUALITY")


@router.get("", response_model=RecommendationResponse)
def recommend(
    use_case: str,
    strategy: str = "balanced",
    top_n: int = 3,
    min_samples: int = 1,
    source: str = "all",
):
    request = RecommendationRequest(
        use_case=use_case,
        strategy=strategy,
        top_n=top_n,
        min_samples=min_samples,
        source=source,  # type: ignore[arg-type]
    )
    try:
        result = _engine.recommend(
            use_case=request.use_case,
            strategy=request.strategy,
            top_n=request.top_n,
            min_samples=request.min_samples,
            source=request.source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    mapped_failure_modes = list(dict.fromkeys(_map_failure_mode(str(m)) for m in result.failure_modes))
    mapped_reliability_label = _map_reliability_label(result.confidence_label)
    evaluation = EvaluationResponse(
        system_health=_map_system_health(result.decision_state),
        evaluation_status=_map_evaluation_status(result.evaluation_status),
        decision=Decision(
            state=_map_decision_state(result.decision_state),
            selected_model=result.best_model or None,
            reason=result.no_recommendation_reason,
        ),
        reliability=DecisionReliability(
            score=result.decision_reliability,
            label=mapped_reliability_label,
        ),
        metrics=EvaluationMetrics(
            score_margin=result.score_margin,
            score_variance=result.score_variance,
            sample_count=result.alternatives[0].sample_count if result.alternatives else 0,
            p95_latency_s=result.alternatives[0].p95_latency if result.alternatives else None,
            consistency_above_threshold=(
                result.alternatives[0].consistency_above_threshold if result.alternatives else None
            ),
        ),
        failure_analysis=FailureAnalysis(
            modes=mapped_failure_modes,
            primary_cause=(mapped_failure_modes[0] if mapped_failure_modes else None),
        ),
        alternatives=[
            ModelAlternative(
                model=a.model,
                score=a.score,
                delta_from_best=a.score_delta_from_best,
                avg_latency_s=a.avg_latency,
                avg_cost_usd=a.avg_cost,
                confidence=a.confidence,
            )
            for a in result.alternatives
        ],
        score_vector=ScoreVector(
            quality_score=result.best_score,
            latency_s=result.alternatives[0].avg_latency if result.alternatives else 0.0,
            cost_usd=result.alternatives[0].avg_cost if result.alternatives else 0.0,
            variance=result.score_variance,
            sample_size=result.alternatives[0].sample_count if result.alternatives else 0,
        ),
        strategy=result.strategy,
        use_case=result.use_case,
        use_case_matched=result.use_case_matched,
    )

    return RecommendationResponse(
        best_model=result.best_model,
        best_score=result.best_score,
        recommendation_available=result.recommendation_available,
        no_recommendation_reason=result.no_recommendation_reason,
        gate_status=result.gate_status,
        gate_threshold=result.gate_threshold,
        gate_triggers=result.gate_triggers,
        validity_status=result.validity_status,
        is_valid=result.is_valid,
        validity_reasons=result.validity_reasons,
        confidence_reasons=result.confidence_reasons,
        alternatives=[
            AlternativeModel(
                model=a.model,
                score=a.score,
                sample_count=a.sample_count,
                avg_latency=a.avg_latency,
                avg_cost=a.avg_cost,
                score_stddev=a.score_stddev,
                score_delta_from_best=a.score_delta_from_best,
                confidence=a.confidence,
                p95_latency=a.p95_latency,
                consistency_above_threshold=a.consistency_above_threshold,
                avg_total_tokens=a.avg_total_tokens,
                avg_cost_per_1k_tokens=a.avg_cost_per_1k_tokens,
                avg_quality_per_1k_tokens=a.avg_quality_per_1k_tokens,
            )
            for a in result.alternatives
        ],
        justification=result.justification,
        evaluation=evaluation,
    )
