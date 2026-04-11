from backend.models.registry import get_model
from backend.agents.hf_agent import HuggingFaceAgent
from backend.rag.retriever import Retriever
from backend.rag.context import RAGContext
from backend.rag.prompt_builder import PromptBuilder
from backend.evaluators.evaluator import Evaluator
from backend.rag.vector_store import VectorStore
import time


class Orchestrator:
    def __init__(self):
        self.agent = HuggingFaceAgent(get_model("small"))
        self.prompt_builder = PromptBuilder()
        self.evaluator = Evaluator()
        self.vector_store = VectorStore()


    def process_task(self, task, metrics=None, model="small", retrieval="rag"):
        # -------------------------
        # 1. Normalise input
        # -------------------------
        if isinstance(task, dict):
            query = task.get("input", "")
            reference = task.get("reference")
        else:
            query = task
            reference = None

        # -------------------------
        # 2. Model switch
        # -------------------------
        self.agent.set_model(get_model(model))

        # -------------------------
        # 3. Retrieval (NOW CORRECT)
        # -------------------------
        retriever = Retriever(store=self.vector_store, mode=retrieval)
        chunks = retriever.search(query)

        # -------------------------
        # 4. Context
        # -------------------------
        context = RAGContext(
            query=query,
            chunks=chunks
        )

        # -------------------------
        # 5. Prompt
        # -------------------------
        prompt = self.prompt_builder.build(context)

        # -------------------------
        # 6. Generation
        # -------------------------
        start_time = time.time()
        output = self.agent.run(prompt)
        latency = time.time() - start_time
        
        
        #---------------------------
        #7. Cost Estimation (mock)
        #---------------------------
        
        cost = self.estimate_cost(output);
        # -------------------------
        # 8. Evaluation
        # -------------------------
        evaluation = {}
        if metrics and reference:
            evaluation = self.evaluator.evaluate(output, reference, metrics)

        # -------------------------
        # 9. Output
        # -------------------------
        return {
            "output": output,
            "evaluation": evaluation,
            "model": model,
            "retrieval": retrieval,
            "latency": latency,
            "cost": cost,
            "context_used": len(chunks) > 0,
            "rag_context": context.to_debug()
        }
        
    def estimate_cost(self, output: str) -> float:
        # simple proxy: tokens ~ words
        tokens = len(output.split())
        return tokens * 0.00001  # mock cost model