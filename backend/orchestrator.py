import time

from backend.models.registry import get_model
from backend.agents.hf_agent import HuggingFaceAgent
from backend.rag.retriever import Retriever
from backend.rag.context import RAGContext
from backend.rag.prompt_builder import PromptBuilder
from backend.evaluators.evaluator import Evaluator
from backend.scoring.registry import get_scorer
from backend.core.types import RunResult


class Orchestrator:
    def __init__(self):
        self.evaluator = Evaluator()
        self.prompt_builder = PromptBuilder()

    def process_task(
        self,
        task,
        model="small",
        retrieval="rag",
        strategy="balanced"
    ):
        # -------------------------
        # 1. Input normalisation
        # -------------------------
        reference = None
        if isinstance(task, dict):
            query = task.get("input", "")
            reference = task.get("reference")

        if isinstance(reference, list):
                reference = " ".join(reference)
        else:
            query = task
            reference = None

        # -------------------------
        # 2. Model (stateless per run)
        # -------------------------
        model_instance = get_model(model)
        agent = HuggingFaceAgent(model_instance)

        # -------------------------
        # 3. Retrieval
        # -------------------------
        retriever = Retriever(mode=retrieval)
        chunks = retriever.search(query)

        context = RAGContext(query=query, chunks=chunks)

        # -------------------------
        # 4. Prompt
        # -------------------------
        prompt = self.prompt_builder.build(context)

        # -------------------------
        # 5. Generation + latency
        # -------------------------
        start = time.time()
        output = agent.run(prompt)
        latency = time.time() - start

        # -------------------------
        # 6. Cost (simple proxy)
        # -------------------------
        cost = len(output.split()) * 0.00001

        # -------------------------
        # 7. Evaluation pipeline
        # -------------------------
        scorer = get_scorer(strategy)

        result = self.evaluator.evaluate(
            output=output,
            reference=reference,
            chunks=chunks,
            scorer=scorer,
            strategy=strategy,
            cost=cost,
            latency=latency
        )

        return RunResult(
            output=output,
            evaluation=result,
            model=model,
            retrieval=retrieval,
            latency=latency,
            cost=cost,
            context_used=len(chunks) > 0,
            rag_context=context.to_debug()
        )