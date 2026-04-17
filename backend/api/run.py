from fastapi import APIRouter

from backend.api.schemas import RunRequest
from backend.orchestrator.orchestrator import Orchestrator
from backend.core.types import ExecutionResponse

router = APIRouter()
orchestrator = Orchestrator()


@router.post("", response_model=ExecutionResponse)
def run_task(request: RunRequest):

    bundle = orchestrator.process_task(
        task=request.input,
        model=request.model,
        strategy=request.strategy
    )

    return ExecutionResponse(
        run=bundle.run,
        runs=None,
        evaluations={"single": bundle.evaluation},
        comparison=None
    )