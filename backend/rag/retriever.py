from backend.rag.context import RetrievedChunk

class Retriever:
    def __init__(self, store, mode="rag"):
        self.store = store
        self.mode = mode

    def search(self, query: str, k: int = 3):
        if self.mode != "rag":
            return []

        results = self.store.search(query, k)

        return [
            RetrievedChunk(
                text=r,       # MUST be string
                score=1.0
            )
            for r in results
        ]
        