# orchestrator.py
import logging
from agents.hf_agent import HuggingFaceAgent
from agents.evaluator_agent import EvaluatorAgent
from rag.embeddings import EmbeddingModel
from metrics.metrics_tracker import MetricsTracker

logging.basicConfig(level=logging.INFO)

class Orchestrator:
    def __init__(self):
        self.agent = HuggingFaceAgent()
        self.evaluator = EvaluatorAgent()
        self.embedding_model = EmbeddingModel()
        self.metrics = MetricsTracker()

    def process_task(self, task: dict):
        logging.info(f"Received task: {task}")

        # Generate embedding
        embedding = self.embedding_model.encode(task["input"])
        logging.info(f"Embedding vector length: {len(embedding)}")

        # Track latency of LLM agent
        output, latency = self.metrics.track_latency(self.agent.run, task["input"])
        logging.info("Agent execution complete")

        # Evaluate output
        evaluation = self.evaluator.evaluate(task["input"], output)

        # Metrics computation (if reference available)
        reference = task.get("reference", "")
        metrics = {"latency": latency, "memory_usage_mb": self.metrics.memory_usage()}

        if reference:
            metrics.update({
                "BERTScore": self.metrics.bert_score(output, reference),
                "METEOR": self.metrics.meteor(output, reference),
                "ROUGE": self.metrics.rouge(output, reference),
                "BLEU": self.metrics.bleu(output, reference),
                "HallucinationRate": self.metrics.hallucination_rate(output, reference.split())
            })

        # Optional: token cost estimation
        if "num_tokens" in task:
            metrics["token_cost_usd"] = self.metrics.estimate_token_cost(task["num_tokens"])

        return {
            "input": task["input"],
            "output": output,
            "embedding_sample": embedding[:5].tolist(),
            "evaluation": evaluation,
            "metrics": metrics
        }