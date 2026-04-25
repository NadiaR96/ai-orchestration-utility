import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

from fastapi import APIRouter, HTTPException, Query

from backend.api.schemas import LeaderboardHistoryQuery, LeaderboardPromptRequest
from backend.core.types import (
    ComparisonResult,
    EvaluationResult,
    LeaderboardEntry,
    LeaderboardResponse,
    RunResult,
)
from backend.evaluators.comparator import Comparator
from backend.orchestrator.orchestrator import Orchestrator
from backend.scoring.registry import SCORERS

router = APIRouter()
orchestrator = Orchestrator()
comparator = Comparator()
LOG_PATH = Path("experiments/logs.jsonl")
SOURCE_EXPERIMENT = "experiment"
SOURCE_LIVE = "live"


def _build_narrative(entry: LeaderboardEntry, sort_strategy: str) -> str:
    score = entry.scores_by_strategy.get(sort_strategy, 0.0)
    rank = entry.ranks_by_strategy.get(sort_strategy, 0)
    message = (
        f"{entry.model}: rank #{rank} on {sort_strategy} "
        f"(score={score:.3f}, latency={entry.run.latency:.3f}s, cost={entry.run.cost:.6f})"
    )

    if entry.trend:
        direction = entry.trend.get("direction", "unknown")
        delta = entry.trend.get("delta_score")
        if isinstance(delta, (float, int)):
            message += f" | trend={direction} ({float(delta):+.3f})"
        else:
            message += f" | trend={direction}"

    return message


def _seed_metrics(run: RunResult, evaluation: EvaluationResult) -> Dict[str, float]:
    metrics = dict(evaluation.metrics or {})
    score = float(evaluation.score)

    metrics.setdefault("bert_score", score)
    metrics.setdefault("faithfulness", score)
    metrics.setdefault("hallucination", max(0.0, 1.0 - score))
    metrics.setdefault("cost_norm", min(max(run.cost, 0.0), 1.0))
    metrics.setdefault("latency_norm", min(max(run.latency, 0.0) / 5.0, 1.0))
    metrics.setdefault("context_used", 1.0 if run.context_used else 0.0)

    return metrics


def _entries_for_strategy(
    entries: List[LeaderboardEntry],
    strategy: str,
) -> List[LeaderboardEntry]:
    return sorted(
        entries,
        key=lambda entry: (
            -entry.scores_by_strategy.get(strategy, 0.0),
            entry.run.latency,
            entry.run.cost,
            entry.model,
        ),
    )


def _rank_and_paginate(
    entries: List[LeaderboardEntry],
    sort_strategy: str,
    page: int,
    page_size: int,
) -> LeaderboardResponse:
    if sort_strategy not in SCORERS:
        raise ValueError(f"Unknown scoring strategy: {sort_strategy}")

    strategy_rankings: Dict[str, ComparisonResult] = {}
    for strategy in SCORERS:
        ordered = _entries_for_strategy(entries, strategy)
        evals = {
            entry.model: EvaluationResult(
                metrics={},
                score=entry.scores_by_strategy.get(strategy, 0.0),
                strategy=strategy,
            )
            for entry in ordered
        }
        comparison = comparator.compare_many(evals, strategy=strategy)
        strategy_rankings[strategy] = comparison

        for rank, model in enumerate(comparison.ranking, start=1):
            for entry in entries:
                if entry.model == model:
                    entry.ranks_by_strategy[strategy] = rank
                    break

    ordered_entries = _entries_for_strategy(entries, sort_strategy)
    total_items = len(ordered_entries)
    start = (page - 1) * page_size
    end = start + page_size
    paged_entries = ordered_entries[start:end]

    for entry in paged_entries:
        entry.narrative = _build_narrative(entry, sort_strategy)

    has_more = end < total_items

    return LeaderboardResponse(
        mode="prompt",
        sort_strategy=sort_strategy,
        page=page,
        page_size=page_size,
        total_items=total_items,
        has_more=has_more,
        next_page=page + 1 if has_more else None,
        items=paged_entries,
        strategy_rankings=strategy_rankings,
    )


def _build_entries_from_runs(runs: Dict[str, RunResult], evaluations: Dict[str, EvaluationResult]) -> List[LeaderboardEntry]:
    entries: List[LeaderboardEntry] = []

    for model, run in runs.items():
        evaluation = evaluations[model]
        metrics = _seed_metrics(run, evaluation)

        scores_by_strategy = {
            strategy: scorer.compute(metrics)
            for strategy, scorer in SCORERS.items()
        }

        entries.append(
            LeaderboardEntry(
                model=model,
                run=run,
                evaluation=evaluation,
                scores_by_strategy=scores_by_strategy,
                ranks_by_strategy={},
                narrative="",
                trend=None,
            )
        )

    return entries


def _parse_models_filter(models: Optional[str]) -> Optional[Set[str]]:
    if not models:
        return None

    parsed = {part.strip() for part in models.split(",") if part.strip()}
    return parsed or None


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00").replace("+00:00", ""))
    except ValueError:
        return None


def _load_latest_entries_from_logs(
    models_filter: Optional[Set[str]] = None,
    source_filter: Optional[str] = None,
    window_hours: Optional[int] = None,
    min_samples: int = 1,
    sort_strategy: Optional[str] = None,
) -> List[LeaderboardEntry]:
    if not LOG_PATH.exists():
        return []

    latest_by_model: Dict[str, dict] = {}
    counts_by_model: Dict[str, int] = {}
    scores_by_model: Dict[str, List[float]] = {}
    min_timestamp = None
    if window_hours is not None:
        min_timestamp = datetime.utcnow() - timedelta(hours=window_hours)

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            model = record.get("model")
            if not model:
                continue

            source = str(record.get("source", SOURCE_EXPERIMENT)).lower()
            if source_filter and source != source_filter:
                continue

            if models_filter and model not in models_filter:
                continue

            if min_timestamp is not None:
                ts = _parse_timestamp(record.get("timestamp"))
                if ts is None or ts < min_timestamp:
                    continue

            counts_by_model[model] = counts_by_model.get(model, 0) + 1

            score = record.get("score")
            record_strategy = str(record.get("strategy", "")).lower()
            if isinstance(score, (int, float)) and (
                sort_strategy is None or record_strategy == sort_strategy.lower()
            ):
                scores_by_model.setdefault(model, []).append(float(score))

            timestamp = str(record.get("timestamp", ""))
            existing = latest_by_model.get(model)
            if existing is None or timestamp > str(existing.get("timestamp", "")):
                latest_by_model[model] = record

    runs: Dict[str, RunResult] = {}
    evaluations: Dict[str, EvaluationResult] = {}
    avg_scores: Dict[str, float] = {}
    sample_counts: Dict[str, int] = {}

    for model, record in latest_by_model.items():
        if counts_by_model.get(model, 0) < min_samples:
            continue

        run = RunResult(
            output=str(record.get("output", "")),
            model=model,
            retrieval=str(record.get("retrieval", "historical")),
            latency=float(record.get("latency", 0.0)),
            cost=float(record.get("cost", 0.0)),
            context_used=bool(record.get("context_used", False)),
            rag_context=dict(record.get("rag_context", {})),
        )

        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}

        evaluation = EvaluationResult(
            metrics=metrics,
            score=float(record.get("score", 0.0)),
            strategy=str(record.get("strategy", "balanced")),
        )

        runs[model] = run
        evaluations[model] = evaluation
        model_scores = scores_by_model.get(model, [])
        if model_scores:
            avg_scores[model] = sum(model_scores) / len(model_scores)
        sample_counts[model] = counts_by_model.get(model, 0)

    entries = _build_entries_from_runs(runs, evaluations)
    for entry in entries:
        entry.latest_score = entry.evaluation.score
        entry.sample_count = sample_counts.get(entry.model, 0)
        if entry.model in avg_scores:
            # Store window avg on the entry for use by live ranking
            entry._window_avg_score = avg_scores[entry.model]  # type: ignore[attr-defined]
    return entries


def _collect_model_score_summary(
    source_filter: str,
    start_time: datetime,
    end_time: datetime,
    models_filter: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, float]]:
    if not LOG_PATH.exists():
        return {}

    scores_by_model: Dict[str, List[float]] = {}

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            model = record.get("model")
            if not model:
                continue

            source = str(record.get("source", SOURCE_EXPERIMENT)).lower()
            if source != source_filter:
                continue

            if models_filter and model not in models_filter:
                continue

            timestamp = _parse_timestamp(record.get("timestamp"))
            if timestamp is None or timestamp < start_time or timestamp >= end_time:
                continue

            score = record.get("score")
            if not isinstance(score, (int, float)):
                continue

            scores_by_model.setdefault(model, []).append(float(score))

    summary: Dict[str, Dict[str, float]] = {}
    for model, scores in scores_by_model.items():
        if not scores:
            continue
        summary[model] = {
            "count": float(len(scores)),
            "avg_score": float(sum(scores) / len(scores)),
        }

    return summary


def _apply_window_avg_scores(
    entries: List[LeaderboardEntry],
    sort_strategy: str,
) -> None:
    """Replace the per-strategy scores used for ranking with the window average score.

    The window average is computed over all runs in the current window for each
    model (stored as ``_window_avg_score`` by the log loader).  When an entry has
    no window average (e.g. only a single out-of-window seed run), the original
    latest-snapshot score is preserved.
    """
    for entry in entries:
        avg = getattr(entry, "_window_avg_score", None)
        if avg is None:
            continue
        # Scale all per-strategy scores proportionally so relative strategy
        # differences are kept, but the sort_strategy drives the primary rank.
        latest = entry.latest_score
        if latest and latest > 0.0:
            ratio = avg / latest
            entry.scores_by_strategy = {
                s: min(max(v * ratio, 0.0), 1.0)
                for s, v in entry.scores_by_strategy.items()
            }
        else:
            # Fallback: replace all strategy scores with the window avg directly.
            entry.scores_by_strategy = {
                s: avg for s in entry.scores_by_strategy
            }


def _attach_live_trends(
    entries: List[LeaderboardEntry],
    window_hours: int,
    models_filter: Optional[Set[str]] = None,
) -> None:
    now = datetime.utcnow()
    current_start = now - timedelta(hours=window_hours)
    previous_start = now - timedelta(hours=window_hours * 2)

    current_summary = _collect_model_score_summary(
        source_filter=SOURCE_LIVE,
        start_time=current_start,
        end_time=now,
        models_filter=models_filter,
    )
    previous_summary = _collect_model_score_summary(
        source_filter=SOURCE_LIVE,
        start_time=previous_start,
        end_time=current_start,
        models_filter=models_filter,
    )

    threshold = 0.01

    for entry in entries:
        current = current_summary.get(entry.model, {})
        previous = previous_summary.get(entry.model, {})

        current_count = int(current.get("count", 0.0))
        previous_count = int(previous.get("count", 0.0))
        current_avg = current.get("avg_score")
        previous_avg = previous.get("avg_score")

        direction = "insufficient_history"
        delta_score = None

        if current_count > 0 and previous_count == 0:
            direction = "new"
        elif current_count > 0 and previous_count > 0 and current_avg is not None and previous_avg is not None:
            delta_score = float(current_avg - previous_avg)
            if delta_score > threshold:
                direction = "up"
            elif delta_score < -threshold:
                direction = "down"
            else:
                direction = "stable"

        entry.trend = {
            "direction": direction,
            "delta_score": delta_score,
            "current_avg_score": current_avg,
            "previous_avg_score": previous_avg,
            "current_samples": current_count,
            "previous_samples": previous_count,
            "window_hours": window_hours,
        }


@router.post("", response_model=LeaderboardResponse)
def leaderboard_prompt(request: LeaderboardPromptRequest):
    if request.aggregation != "latest":
        raise HTTPException(status_code=422, detail="Only aggregation=latest is supported")

    sort_strategy = request.sort_strategy.lower()
    if sort_strategy not in SCORERS:
        raise HTTPException(status_code=422, detail=f"Unknown scoring strategy: {sort_strategy}")

    runs: Dict[str, RunResult] = {}
    evaluations: Dict[str, EvaluationResult] = {}

    for model in request.models:
        bundle = orchestrator.process_task(
            task={"input": request.input, "reference": request.reference},
            model=model,
            retrieval=request.retrieval,
            strategy=sort_strategy,
        )
        runs[model] = bundle.run
        evaluations[model] = bundle.evaluation

    entries = _build_entries_from_runs(runs, evaluations)
    response = _rank_and_paginate(entries, sort_strategy, request.page, request.page_size)
    response.mode = "prompt"
    return response


@router.get("", response_model=LeaderboardResponse)
def leaderboard_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    sort_strategy: str = Query(default="balanced"),
    aggregation: str = Query(default="latest"),
    models: Optional[str] = Query(default=None),
):
    if aggregation != "latest":
        raise HTTPException(status_code=422, detail="Only aggregation=latest is supported")

    query = LeaderboardHistoryQuery(
        page=page,
        page_size=page_size,
        sort_strategy=sort_strategy,
        aggregation=aggregation,
        models=list(_parse_models_filter(models) or []),
    )

    entries = _load_latest_entries_from_logs(
        models_filter=_parse_models_filter(models),
        source_filter=SOURCE_EXPERIMENT,
        min_samples=1,
    )
    response = _rank_and_paginate(entries, query.sort_strategy.lower(), query.page, query.page_size)
    response.mode = "historical"
    return response


@router.get("/experiments", response_model=LeaderboardResponse)
def leaderboard_experiments(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    sort_strategy: str = Query(default="balanced"),
    aggregation: str = Query(default="latest"),
    models: Optional[str] = Query(default=None),
):
    if aggregation != "latest":
        raise HTTPException(status_code=422, detail="Only aggregation=latest is supported")

    query = LeaderboardHistoryQuery(
        page=page,
        page_size=page_size,
        sort_strategy=sort_strategy,
        aggregation=aggregation,
        models=list(_parse_models_filter(models) or []),
    )

    entries = _load_latest_entries_from_logs(
        models_filter=_parse_models_filter(models),
        source_filter=SOURCE_EXPERIMENT,
        min_samples=1,
    )
    response = _rank_and_paginate(entries, query.sort_strategy.lower(), query.page, query.page_size)
    response.mode = "experiments"
    return response


@router.get("/live", response_model=LeaderboardResponse)
def leaderboard_live(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    sort_strategy: str = Query(default="balanced"),
    window_hours: int = Query(default=24, ge=1, le=168),
    min_samples: int = Query(default=1, ge=1, le=1000),
    models: Optional[str] = Query(default=None),
    ranking_basis: str = Query(default="window_avg", pattern="^(latest|window_avg)$"),
):
    entries = _load_latest_entries_from_logs(
        models_filter=_parse_models_filter(models),
        source_filter=SOURCE_LIVE,
        window_hours=window_hours,
        min_samples=min_samples,
        sort_strategy=sort_strategy.lower() if ranking_basis == "window_avg" else None,
    )

    if ranking_basis == "window_avg":
        _apply_window_avg_scores(entries, sort_strategy.lower())

    _attach_live_trends(
        entries=entries,
        window_hours=window_hours,
        models_filter=_parse_models_filter(models),
    )

    response = _rank_and_paginate(entries, sort_strategy.lower(), page, page_size)
    response.mode = "live"
    return response
