from transformers import pipeline


class HuggingFaceAgent:
    def __init__(self, model_name="gpt2"):
        self.generator = pipeline("text-generation", model=model_name)

    def run(self, input_text: str) -> str:
        response = self.generator(
            input_text,
            max_length=100,
            num_return_sequences=1
        )
        return response[0]["generated_text"]