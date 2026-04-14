from typing import List, Dict, Any
from backend.experiments.results import ExperimentResult
from backend.orchestrator.orchestrator import Orchestrator


class ExperimentRunner:
    def __init__(self):
        self.orch = Orchestrator()

    def run_single(self, prompt: str, config: Dict[str, Any], reference: str | None = None):
        result = self.orch.process_task(
            task={
                "input": prompt,
                "reference": reference
            },
            model=config.get("model", "small"),
            retrieval=config.get("retrieval", "rag"),
            metrics=config.get("metrics")
        )

        return ExperimentResult(
            output=result["output"],
            model=result["model"],
            retrieval=result["retrieval"],
            metrics=result["evaluation"],
            context=result["rag_context"]
        )

    def run_batch(
        self,
        prompt: str,
        configs: List[Dict[str, Any]],
        reference: str | None = None
    ) -> List[ExperimentResult]:

        results = []

        for config in configs:
            exp = self.run_single(prompt, config, reference)
            results.append(exp)

        return results

    def compare_pair(
        self,
        prompt: str,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
        reference: str | None = None,
        compare_fn=None
    ):
        exp_a = self.run_single(prompt, config_a, reference)
        exp_b = self.run_single(prompt, config_b, reference)

        comparison = None
        if compare_fn:
            comparison = compare_fn(exp_a, exp_b)

        return {
            "A": exp_a.__dict__,
            "B": exp_b.__dict__,
            "comparison": comparison
        }