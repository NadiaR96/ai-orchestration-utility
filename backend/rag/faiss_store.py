from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class FAISSStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)

        # embedding dimension for MiniLM
        self.dim = 384

        # FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(self.dim)

        # stored documents aligned with vector index
        self.documents: list[str] = []

    def _to_vector(self, texts):
        """
        Normalises embeddings into FAISS-compatible format:
        - float32
        - 2D shape (n, dim)
        """
        vectors = self.embedder.encode(texts)
        vectors = np.asarray(vectors, dtype="float32")

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        return vectors

    def add_documents(self, docs: list[str]):
        if not docs:
            return

        embeddings = self._to_vector(docs)

        # safety check: dimension match
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim mismatch: expected {self.dim}, got {embeddings.shape[1]}"
            )

        self.index.add(embeddings) #type: ignore
        self.documents.extend(docs)

    def search(self, query: str, k: int = 3):
        if len(self.documents) == 0:
            return []

        query_vec = self._to_vector([query])

        distances, indices = self.index.search(query_vec, k) # type: ignore

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1:
                continue
            if 0 <= i < len(self.documents):
                results.append(
                    (self.documents[i], float(dist))
                )

        return results

    def reset(self):
        """
        Clears index + documents (useful for testing)
        """
        self.index = faiss.IndexFlatL2(self.dim)
        self.documents = []