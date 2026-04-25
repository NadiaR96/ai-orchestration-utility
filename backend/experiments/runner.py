import uuid
from backend.orchestrator.orchestrator import Orchestrator
from backend.evaluators.comparator import Comparator
from backend.experiments.experiment import ExperimentConfig, ExperimentResult as ExperimentRunSummary
from backend.experiments.results import ExperimentResult
from backend.experiments.tracker import ExperimentTracker
from backend.core.types import EvaluationResult, RunBundle, RunResult


class ExperimentRunner:

    def __init__(self):
        self.orchestrator = Orchestrator()
        self.comparator = Comparator()
        self.tracker = ExperimentTracker()

    def run(self, config: ExperimentConfig):

        experiment_id = config.name or str(uuid.uuid4())
        use_case = (getattr(config, "use_case", None) or config.name or "").strip() or None

        run_matrix = {}
        comparisons = []

        # -------------------------
        # 1. Run all inputs × models
        # -------------------------
        runs_per_input = max(1, int(getattr(config, "runs_per_input", 1) or 1))

        for input_text in config.inputs:

            runs = {}

            for model in config.models:
                model_bundles = []

                for run_iteration in range(runs_per_input):
                    bundle = self.orchestrator.process_task(
                        task=input_text,
                        model=model,
                        strategy=config.strategy
                    )

                    model_bundles.append(bundle)

                    # -------------------------
                    # 2. LOG EACH RUN
                    # -------------------------
                    self.tracker.log(
                        {
                            "source": "experiment",
                            "use_case": use_case,
                            "input": input_text,
                            "model": model,
                            "output": bundle.run.output,
                            "score": bundle.evaluation.score,
                            "strategy": config.strategy,
                            "latency": bundle.run.latency,
                            "cost": bundle.run.cost,
                            "prompt_tokens": bundle.run.prompt_tokens,
                            "output_tokens": bundle.run.output_tokens,
                            "total_tokens": bundle.run.total_tokens,
                            "cost_per_1k_tokens": bundle.run.cost_per_1k_tokens,
                            "metrics": bundle.evaluation.metrics,
                            "run_iteration": run_iteration + 1,
                            "runs_per_input": runs_per_input,
                        },
                        experiment_id=experiment_id
                    )

                runs[model] = self._aggregate_model_runs(model_bundles, model=model, strategy=config.strategy)

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

    def _aggregate_model_runs(self, bundles, model: str, strategy: str) -> RunBundle:
        if len(bundles) == 1:
            return bundles[0]

        total = float(len(bundles))

        avg_score = sum(float(b.evaluation.score) for b in bundles) / total
        avg_latency = sum(float(b.run.latency) for b in bundles) / total
        avg_cost = sum(float(b.run.cost) for b in bundles) / total
        avg_prompt_tokens = sum(float(b.run.prompt_tokens) for b in bundles) / total
        avg_output_tokens = sum(float(b.run.output_tokens) for b in bundles) / total
        avg_total_tokens = sum(float(b.run.total_tokens) for b in bundles) / total
        avg_cost_per_1k_tokens = sum(float(b.run.cost_per_1k_tokens) for b in bundles) / total

        # Keep representative output/context from highest-scoring run for readability.
        best_bundle = max(bundles, key=lambda b: float(b.evaluation.score))

        metric_keys = set()
        for b in bundles:
            metric_keys.update((b.evaluation.metrics or {}).keys())

        aggregated_metrics = {}
        for key in metric_keys:
            values = []
            for b in bundles:
                value = (b.evaluation.metrics or {}).get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            if values:
                aggregated_metrics[key] = sum(values) / float(len(values))
            else:
                aggregated_metrics[key] = (best_bundle.evaluation.metrics or {}).get(key, 0.0)

        run = RunResult(
            output=best_bundle.run.output,
            model=model,
            retrieval=best_bundle.run.retrieval,
            latency=avg_latency,
            cost=avg_cost,
            context_used=best_bundle.run.context_used,
            rag_context=best_bundle.run.rag_context,
            prompt_tokens=int(round(avg_prompt_tokens)),
            output_tokens=int(round(avg_output_tokens)),
            total_tokens=int(round(avg_total_tokens)),
            cost_per_1k_tokens=avg_cost_per_1k_tokens,
        )
        evaluation = EvaluationResult(
            metrics=aggregated_metrics,
            score=avg_score,
            strategy=strategy,
        )
        return RunBundle(run=run, evaluation=evaluation)

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