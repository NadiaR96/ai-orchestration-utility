from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from backend.experiments.runner import ExperimentRunner
from backend.experiments.experiment import ExperimentConfig

router = APIRouter()
runner = ExperimentRunner()


@router.post("/experiment")
async def run_experiment(payload: dict):

    config = ExperimentConfig(**payload)

    result = await run_in_threadpool(runner.run, config)

    return result