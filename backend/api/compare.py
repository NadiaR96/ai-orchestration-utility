from fastapi import APIRouter

from backend.api.schemas import CompareRequest
from backend.orchestrator.orchestrator import Orchestrator
from backend.evaluators.comparator import Comparator
from backend.core.types import ExecutionResponse

router = APIRouter()


@router.post("", response_model=ExecutionResponse)
def compare_v2(request: CompareRequest):

    orchestrator = Orchestrator()
    comparator = Comparator()

    runs = {}
    evaluations = {}

    # -------------------------
    # 1. Run models
    # -------------------------
    for model in request.models:

        bundle = orchestrator.process_task(
            task=request.input,
            model=model,
            strategy=request.strategy
        )

        runs[model] = bundle.run
        evaluations[model] = bundle.evaluation

    # -------------------------
    # 2. Compare
    # -------------------------
    comparison = None

    if len(evaluations) > 1:
        comparison = comparator.compare_many(evaluations)

    # -------------------------
    # 3. Return typed response
    # -------------------------
    return ExecutionResponse(
        run=None,
        runs=runs,
        evaluations=evaluations,
        comparison=comparison
    )