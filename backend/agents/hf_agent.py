from transformers import pipeline
from backend.agents.base import BaseAgent


class HuggingFaceAgent(BaseAgent):
    def __init__(self, model_name: str):
        super().__init__()
        self.set_model(model_name)

    def set_model(self, model_name: str):
        self.model_name = model_name
        self.generator = pipeline(
            "text-generation",
            model=model_name
        )

    def run(self, prompt: str) -> str:
        result = self.generator(
            prompt,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2
        )

        generated_text = result[0]["generated_text"]

        # IMPORTANT: strip prompt if model echoes it (prevents bad evaluation)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text