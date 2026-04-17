import uuid
from backend.orchestrator.orchestrator import Orchestrator
from backend.evaluators.comparator import Comparator
from backend.experiments.experiment import ExperimentConfig, ExperimentResult as ExperimentRunSummary
from backend.experiments.results import ExperimentResult
from backend.experiments.tracker import ExperimentTracker


class ExperimentRunner:

    def __init__(self):
        self.orchestrator = Orchestrator()
        self.comparator = Comparator()
        self.tracker = ExperimentTracker()

    def run(self, config: ExperimentConfig):

        experiment_id = config.name or str(uuid.uuid4())

        run_matrix = {}
        comparisons = []

        # -------------------------
        # 1. Run all inputs × models
        # -------------------------
        for input_text in config.inputs:

            runs = {}

            for model in config.models:

                bundle = self.orchestrator.process_task(
                    task=input_text,
                    model=model,
                    strategy=config.strategy
                )

                runs[model] = bundle

                # -------------------------
                # 2. LOG EACH RUN 
                # -------------------------
                self.tracker.log(
                    {
                        "input": input_text,
                        "model": model,
                        "output": bundle.run.output,
                        "score": bundle.evaluation.score,
                        "strategy": config.strategy,
                        "latency": bundle.run.latency,
                        "cost": bundle.run.cost,
                    },
                    experiment_id=experiment_id
                )

            run_matrix[input_text] = runs

            # -------------------------
            # 3. Compare per input
            # -------------------------
            comparison = self.comparator.compare_many(
                {m: r.evaluation for m, r in runs.items()},
                strategy=config.strategy
            )

            comparisons.append({
                "input": input_text,
                "comparison": comparison
            })

        # -------------------------
        # 4. Summary
        # -------------------------
        summary = self._build_summary(comparisons)

        return ExperimentRunSummary(
            name=experiment_id,
            comparisons=comparisons,
            run_matrix=run_matrix,
            summary=summary
        )

    def run_single(self, prompt, config, reference=None):
        model = config.get("model", "small")
        retrieval = config.get("retrieval", "rag")
        metrics = config.get("metrics")

        result = self.orchestrator.process_task(
            task={"input": prompt, "reference": reference},
            model=model,
            retrieval=retrieval,
        )

        return self._to_result(result, model, retrieval)

    def run_batch(self, prompt, configs, reference=None):
        return [self.run_single(prompt, config, reference=reference) for config in configs]

    def compare_pair(self, prompt, config_a, config_b, reference=None, compare_fn=None):
        result_a = self.run_single(prompt, config_a, reference=reference)
        result_b = self.run_single(prompt, config_b, reference=reference)

        comparison = compare_fn(result_a, result_b) if compare_fn else None

        return {
            "A": result_a.__dict__,
            "B": result_b.__dict__,
            "comparison": comparison,
        }

    def _build_summary(self, comparisons):
        wins = {}

        for c in comparisons:
            winner = c["comparison"].winner
            wins[winner] = wins.get(winner, 0) + 1

        return {
            "win_counts": wins
        }

    def _to_result(self, result, default_model, default_retrieval):
        if isinstance(result, dict):
            return ExperimentResult(
                output=result.get("output", ""),
                model=result.get("model", default_model),
                retrieval=result.get("retrieval", default_retrieval),
                metrics=result.get("evaluation", {}),
                rag_context=result.get("rag_context", {}),
            )

        return ExperimentResult(
            output=result.run.output,
            model=result.run.model,
            retrieval=result.run.retrieval,
            metrics={"score": result.evaluation.score, **result.evaluation.metrics},
            rag_context=result.run.rag_context,
        )