from dataclasses import dataclass
from typing import Any


@dataclass
class ExperimentResult:
    output: str
    model: str
    retrieval: str
    metrics: dict[str, Any]
    rag_context: dict[str, Any]