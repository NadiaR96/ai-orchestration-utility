from backend.rag.context import RetrievedChunk
from backend.rag.null_retriever import NullRetriever

class Retriever:
    def __init__(self, store=None, mode="rag"):
        self.store = store or NullRetriever()
        self.mode = mode

    def search(self, query: str, k: int = 3):
        if self.mode != "rag":
            return []

        results = self.store.search(query, k)

        chunks = []
        for result in results:
            if isinstance(result, tuple) and len(result) >= 2:
                text = str(result[0])
                score = float(result[1])
            else:
                text = str(result)
                score = 1.0

            chunks.append(RetrievedChunk(text=text, score=score))

        return chunks
        