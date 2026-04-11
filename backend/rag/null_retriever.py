class NullRetriever:
    def search(self, query: str, k: int = 3):
        base = [
            "RAG stands for Retrieval Augmented Generation.",
            "It improves LLM responses by injecting external context.",
            "Vector databases store embeddings for semantic search.",
            "FAISS is a similarity search library.",
        ]

        # always include query grounding
        return [f"Query context: {query}"] + base[: k - 1]