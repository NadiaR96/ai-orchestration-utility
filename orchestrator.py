import logging
from agents.hf_agent import HuggingFaceAgent
from rag.embeddings import EmbeddingModel

logging.basicConfig(level=logging.INFO)


class Orchestrator:
    def __init__(self):
        self.agent = HuggingFaceAgent()
        self.embedding_model = EmbeddingModel()

    def process_task(self, task: dict):
        logging.info(f"Received task: {task}")

        # Step 1: Generate embedding (RAG prep)
        embedding = self.embedding_model.encode(task["input"])

        logging.info(f"Generated embedding vector length: {len(embedding)}")

        # Step 2: Run agent
        result = self.agent.run(task["input"])

        logging.info("Agent execution complete")

        return {
            "input": task["input"],
            "output": result,
            "embedding_sample": embedding[:5].tolist()
        }