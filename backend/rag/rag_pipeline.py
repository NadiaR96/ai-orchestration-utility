from backend.rag.context import RAGContext


class RAGPipeline:
    def __init__(self, retriever, prompt_builder):
        self.retriever = retriever
        self.prompt_builder = prompt_builder

    def run(self, query: str):
        chunks = self.retriever.search(query)
        context = RAGContext(query=query, chunks=chunks)
        prompt = self.prompt_builder.build(context)

        return {
            "prompt": prompt,
            "chunks": chunks
        }