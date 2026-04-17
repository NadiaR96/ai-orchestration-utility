class PromptBuilder:
    def build(self, context) -> str:
        chunks = "\n".join(
            [
                f"[{i+1}] {c.text}"
                for i, c in enumerate(context.chunks)
            ]
        )

        return f"""
You are a retrieval-augmented assistant.

Use the context as your primary source of truth.

If the context is incomplete, you may reason carefully, but do not hallucinate facts.

Answer in clear, natural language.
Be concise but complete.
Prefer explanations over definitions. 


---

CONTEXT:
{chunks}

---

QUESTION:
{context.query}

---

ANSWER:
"""