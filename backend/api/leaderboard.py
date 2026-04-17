import json
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


def _build_narrative(entry: LeaderboardEntry, sort_strategy: str) -> str:
    score = entry.scores_by_strategy.get(sort_strategy, 0.0)
    rank = entry.ranks_by_strategy.get(sort_strategy, 0)
    return (
        f"{entry.model}: rank #{rank} on {sort_strategy} "
        f"(score={score:.3f}, latency={entry.run.latency:.3f}s, cost={entry.run.cost:.6f})"
    )


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
            )
        )

    return entries


def _parse_models_filter(models: Optional[str]) -> Optional[Set[str]]:
    if not models:
        return None

    parsed = {part.strip() for part in models.split(",") if part.strip()}
    return parsed or None


def _load_latest_entries_from_logs(models_filter: Optional[Set[str]] = None) -> List[LeaderboardEntry]:
    if not LOG_PATH.exists():
        return []

    latest_by_model: Dict[str, dict] = {}

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            model = record.get("model")
            if not model:
                continue
            if models_filter and model not in models_filter:
                continue

            timestamp = str(record.get("timestamp", ""))
            existing = latest_by_model.get(model)
            if existing is None or timestamp > str(existing.get("timestamp", "")):
                latest_by_model[model] = record

    runs: Dict[str, RunResult] = {}
    evaluations: Dict[str, EvaluationResult] = {}

    for model, record in latest_by_model.items():
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

    return _build_entries_from_runs(runs, evaluations)


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

    entries = _load_latest_entries_from_logs(_parse_models_filter(models))
    response = _rank_and_paginate(entries, query.sort_strategy.lower(), query.page, query.page_size)
    response.mode = "historical"
    return response
