import time

from backend.models.registry import get_model
from backend.agents.hf_agent import HuggingFaceAgent
from backend.rag.retriever import Retriever
from backend.rag.context import RAGContext
from backend.rag.prompt_builder import PromptBuilder
from backend.evaluators.evaluator import Evaluator
from backend.core.types import RunResult,RunBundle


class Orchestrator:
    def __init__(self):
        self.evaluator = Evaluator()
        self.prompt_builder = PromptBuilder()

    def process_task(self, task, model="small", retrieval="rag", strategy="balanced", metrics=None):

        # -------------------------
        # 1. Parse input
        # -------------------------
        if isinstance(task, dict):
            query = task.get("input", "")
            reference = task.get("reference")
        else:
            query = task
            reference = None

        if isinstance(reference, list):
            reference = " ".join(reference)

        # -------------------------
        # 2. Model
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
        # 5. Generation
        # -------------------------
        start = time.time()
        output = agent.run(prompt)
        latency = time.time() - start

        # -------------------------
        # 6. Cost
        # -------------------------
        cost = len(output.split()) * 0.00001

        # -------------------------
        # 7. Evaluation (NO scorer registry here)
        # -------------------------
        evaluation = self.evaluator.evaluate(
            output=output,
            reference=reference,
            chunks=chunks,
            strategy=strategy,
            cost=cost,
            latency=latency
        )

        # -------------------------
        # 8. Return PURE RunResult (no evaluation inside)
        # -------------------------
        run = RunResult(
            output=output,
            model=model,
            retrieval=retrieval,
            latency=latency,
            cost=cost,
            context_used=len(chunks) > 0,
            rag_context=context.to_debug()
        )

        return RunBundle(run=run, evaluation=evaluation)