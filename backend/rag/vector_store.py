from backend.rag.faiss_store import FAISSStore

class VectorStore:
    """
    Single persistent wrapper around FAISSStore.
    This prevents re-initialisation per request.
    """

    def __init__(self):
        self.store = FAISSStore()

        # optional bootstrap knowledge (TEMP for dev)
        self._bootstrap()

    def _bootstrap(self):
        docs = [
            """RAG (Retrieval Augmented Generation) is a technique where an LLM retrieves relevant external documents before generating a response.

Instead of relying only on its internal training data, the model queries a vector database (e.g., FAISS) to fetch semantically similar passages.

This improves factual accuracy, reduces hallucination, and allows the system to use up-to-date or domain-specific knowledge.

A typical RAG pipeline includes:
1. Query embedding
2. Vector similarity search
3. Context injection into prompt
4. Response generation"""
        ]
        self.store.add_documents(docs)

    def add(self, docs: list[str]):
        self.store.add_documents(docs)

    def search(self, query: str, k: int = 3):
        return self.store.search(query, k)