from fastapi import APIRouter

from backend.api.schemas import RunRequest
from backend.orchestrator.orchestrator import Orchestrator
from backend.core.types import ExecutionResponse
from backend.experiments.tracker import ExperimentTracker

router = APIRouter()
orchestrator = Orchestrator()
tracker = ExperimentTracker()


@router.post("", response_model=ExecutionResponse)
def run_task(request: RunRequest):

    task = {"input": request.input, "reference": request.reference} if request.reference else request.input
    bundle = orchestrator.process_task(
        task=task,
        model=request.model,
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

    return ExecutionResponse(
        run=bundle.run,
        runs=None,
        evaluations={"single": bundle.evaluation},
        comparison=None
    )