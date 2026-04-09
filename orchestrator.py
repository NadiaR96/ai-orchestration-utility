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

        task_input = task.get("input", task.get("prompt", ""))
        reference = task.get("reference", task_input)

        # Generate embedding
        embedding = self.embedding_model.encode(task_input)
        logging.info(f"Embedding vector length: {len(embedding)}")

        # Track latency of agent
        output, latency = self.metrics.track_latency(self.agent.run, task_input)

        # Evaluate output
        evaluation = self.evaluator.evaluate(output, reference)

        # Metrics
        metrics = {
            "latency": latency,
            "memory_usage_mb": self.metrics.memory_usage()
        }

        if reference:
            ref_tokens = reference.split()
            metrics.update({
                "BERTScore": self.metrics.bert_score([output], [reference]),
                "METEOR": self.metrics.meteor(output, reference),
                "ROUGE": self.metrics.rouge(output, reference),
                "BLEU": self.metrics.bleu(output, reference),
                "HallucinationRate": self.metrics.hallucination_rate(output, ref_tokens),
                "F1_Precision_Recall": self.metrics.f1_precision_recall(ref_tokens, output.split()),
                "Diversity": self.metrics.diversity_score(output),
                "Coverage": self.metrics.coverage_score(output, ref_tokens)
            })

        # Token cost if available
        if "num_tokens" in task:
            metrics["token_cost_usd"] = self.metrics.estimate_token_cost(task["num_tokens"])

        result = {
            "input": task_input,
            "output": output,
            "embedding_sample": embedding[:5].tolist(),
            "evaluation": evaluation,
            "metrics": metrics
        }
        result.update(evaluation)
        return result