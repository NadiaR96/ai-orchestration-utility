from fastapi import APIRouter

from backend.experiments.runner import ExperimentRunner
from backend.experiments.experiment import ExperimentConfig

router = APIRouter()
runner = ExperimentRunner()


@router.post("/experiment")
def run_experiment(payload: dict):

    config = ExperimentConfig(**payload)

    result = runner.run(config)

    return result