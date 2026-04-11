class RAGPipeline:
    def __init__(self, retriever, prompt_builder):
        self.retriever = retriever
        self.prompt_builder = prompt_builder

    def run(self, query: str):
        chunks = self.retriever.search(query)
        prompt = self.prompt_builder.build(query, chunks)

        return {
            "prompt": prompt,
            "chunks": chunks
        }