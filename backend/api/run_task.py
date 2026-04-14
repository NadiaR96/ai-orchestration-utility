from fastapi import APIRouter
from backend.orchestrator import Orchestrator

router = APIRouter()
orchestrator = Orchestrator()


@router.post("")
def run_task(request: dict):
    return orchestrator.process_task(request)