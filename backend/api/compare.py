from fastapi import APIRouter
from backend.orchestrator import Orchestrator
from backend.evaluators.comparator import Comparator

router = APIRouter()

orchestrator = Orchestrator()
comparator = Comparator()


@router.post("")
def compare(request: dict):

    result_a = orchestrator.process_task(
        {"input": request["input"], "reference": request.get("reference")},
        model=request.get("model_a", "small"),
        retrieval=request.get("retrieval", "rag"),
        strategy=request.get("strategy", "balanced")
    )

    result_b = orchestrator.process_task(
        {"input": request["input"], "reference": request.get("reference")},
        model=request.get("model_b", "google/flan-t5-base"),
        retrieval=request.get("retrieval", "rag"),
        strategy=request.get("strategy", "balanced")
    )

    comparison = comparator.compare(
        result_a.evaluation,
        result_b.evaluation
    )

    return {
        "A": result_a,
        "B": result_b,
        "comparison": comparison
    }