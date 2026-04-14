from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class RunResult:
    output: str
    evaluation: Any  # EvaluationResult (avoids circular imports)
    model: str
    retrieval: str
    latency: float
    cost: float
    context_used: bool
    rag_context: Dict[str, Any]
    
@dataclass
class EvaluationResult:
    metrics: Dict[str, Any]
    score: float
    strategy: str