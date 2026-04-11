from fastapi import FastAPI
from pydantic import BaseModel
from backend.orchestrator import Orchestrator
from backend.experiments.runner import ExperimentRunner
from backend.evaluators.comparator import compare_results

app = FastAPI()
orch = Orchestrator()
runner = ExperimentRunner()


class TaskRequest(BaseModel):
    prompt: str
    reference: str | None = None
    model: str = "small"
    retrieval: str = "rag"
    metrics: list[str] | None = None


class CompareRequest(BaseModel):
    prompt: str
    reference: str | None = None
    config_a: dict
    config_b: dict
    strategy: str = "balanced"


@app.get("/")
def root():
    return {"message": "AI Orchestration API running"}


@app.post("/run-task")
def run_task(req: TaskRequest):
    return orch.process_task(
        task={
            "input": req.prompt,
            "reference": req.reference
        },
        model=req.model,
        retrieval=req.retrieval,
        metrics=req.metrics
    )


@app.post("/compare")
def compare(req: CompareRequest):
    result_a = orch.process_task(
        task={"input": req.prompt, "reference": req.reference},
        retrieval=req.config_a.get("retrieval", "rag"),
        model=req.config_a["model"]
    )

    result_b = orch.process_task(
        task={"input": req.prompt, "reference": req.reference},
        retrieval=req.config_b.get("retrieval", "rag"),
        model=req.config_b["model"]
    )

    comparison = compare_results(result_a, result_b, req.strategy)

    return {
        "A": result_a,
        "B": result_b,
        "comparison": comparison
    }