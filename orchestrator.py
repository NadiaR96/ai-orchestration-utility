import logging
from agents.hf_agent import HuggingFaceAgent
from agents.evaluator_agent import EvaluatorAgent
from rag.embeddings import EmbeddingModel

logging.basicConfig(level=logging.INFO)

class Orchestrator:
    def __init__(self):
        self.agent = HuggingFaceAgent()
        self.evaluator = EvaluatorAgent()
        self.embedding_model = EmbeddingModel()

    def process_task(self, task: dict):
        logging.info(f"Received task: {task}")

        # Generate embedding (RAG prep)
        embedding = self.embedding_model.encode(task["input"])
        logging.info(f"Generated embedding vector length: {len(embedding)}")

        # Run LLM agent
        agent_output = self.agent.run(task["input"])
        logging.info("Agent execution complete")

        # Evaluate output
        evaluation = self.evaluator.evaluate(task["input"], agent_output)
        logging.info(f"Evaluation complete: {evaluation}")

        return {
            "input": task["input"],
            "output": agent_output,
            "embedding_sample": embedding[:5].tolist(),
            "evaluation": evaluation
        }