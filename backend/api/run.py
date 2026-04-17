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

    bundle = orchestrator.process_task(
        task=request.input,
        model=request.model,
        strategy=request.strategy
    )

    tracker.log(
        {
            "source": "live",
            "input": request.input,
            "model": bundle.run.model,
            "output": bundle.run.output,
            "score": bundle.evaluation.score,
            "strategy": bundle.evaluation.strategy,
            "latency": bundle.run.latency,
            "cost": bundle.run.cost,
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