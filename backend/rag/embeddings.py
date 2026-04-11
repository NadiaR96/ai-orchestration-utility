from sentence_transformers import SentenceTransformer

class Embeddings:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def encode(self, texts):
        return self.model.encode(texts)