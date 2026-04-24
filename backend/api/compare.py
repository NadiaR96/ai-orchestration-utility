from fastapi import APIRouter

from backend.api.schemas import CompareRequest
from backend.orchestrator.orchestrator import Orchestrator
from backend.evaluators.comparator import Comparator
from backend.core.types import ExecutionResponse
from backend.experiments.tracker import ExperimentTracker

router = APIRouter()
tracker = ExperimentTracker()


@router.post("", response_model=ExecutionResponse)
def compare_v2(request: CompareRequest):

    orchestrator = Orchestrator()
    comparator = Comparator()

    runs = {}
    evaluations = {}

    # -------------------------
    # 1. Run models
    # -------------------------
    task = {"input": request.input, "reference": request.reference} if request.reference else request.input

    for model in request.models:

        bundle = orchestrator.process_task(
            task=task,
            model=model,
            retrieval=request.retrieval,
            strategy=request.strategy
        )

        tracker.log(
            {
                "source": "live",
                "use_case": request.use_case,
                "input": request.input,
                "model": bundle.run.model,
                "output": bundle.run.output,
                "score": bundle.evaluation.score,
                "strategy": bundle.evaluation.strategy,
                "latency": bundle.run.latency,
                "cost": bundle.run.cost,
                "prompt_tokens": bundle.run.prompt_tokens,
                "output_tokens": bundle.run.output_tokens,
                "total_tokens": bundle.run.total_tokens,
                "cost_per_1k_tokens": bundle.run.cost_per_1k_tokens,
                "retrieval": bundle.run.retrieval,
                "context_used": bundle.run.context_used,
                "rag_context": bundle.run.rag_context,
                "metrics": bundle.evaluation.metrics,
            }
        )

        runs[model] = bundle.run
        evaluations[model] = bundle.evaluation

    # -------------------------
    # 2. Compare
    # -------------------------
    comparison = None

    if len(evaluations) > 1:
        comparison = comparator.compare_many(evaluations, strategy=request.strategy)

    # -------------------------
    # 3. Return typed response
    # -------------------------
    return ExecutionResponse(
        run=None,
        runs=runs,
        evaluations=evaluations,
        comparison=comparison
    )