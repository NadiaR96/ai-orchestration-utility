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
        self.generation_config = self.generator.model.generation_config
        self.generation_config.max_new_tokens = 80
        self.generation_config.do_sample = True
        self.generation_config.temperature = 0.7
        self.generation_config.top_p = 0.9
        self.generation_config.repetition_penalty = 1.2
        self.generation_config.max_length = None

        tokenizer = getattr(self.generator, "tokenizer", None)
        if tokenizer is not None and tokenizer.eos_token_id is not None:
            self.generation_config.pad_token_id = tokenizer.eos_token_id

    def run(self, prompt: str) -> str:
        result = self.generator(
            prompt,
            generation_config=self.generation_config,
            return_full_text=True
        )

        generated_text = result[0]["generated_text"]

        # IMPORTANT: strip prompt if model echoes it (prevents bad evaluation)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        return generated_text