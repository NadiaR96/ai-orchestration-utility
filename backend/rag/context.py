from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class RetrievedChunk:
    text: str
    score: float
    source: str | None = None


@dataclass
class RAGContext:
    query: str
    chunks: List[RetrievedChunk]

    def to_text(self) -> str:
        return "\n\n".join(
            [f"[{i+1}] {c.text}" for i, c in enumerate(self.chunks)]
        )

    def to_debug(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "chunks": [
                {
                    "text": c.text,
                    "score": c.score,
                    "source": c.source
                }
                for c in self.chunks
            ]
        }