from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class RunResult:
    output: str
    model: str
    retrieval: str
    latency: float
    cost: float
    context_used: bool
    rag_context: Dict[str, Any]
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_per_1k_tokens: float = 0.0
    
@dataclass
class EvaluationResult:
    metrics: Dict[str, Any]
    score: float
    strategy: str

@dataclass
class RunBundle:
    run: RunResult
    evaluation: EvaluationResult
    
@dataclass
class ComparisonResult:
    winner: Optional[str]
    ranking: List[str]
    score_breakdown: Dict[str, float]
    strategy: str
    tied_winners: List[str] = field(default_factory=list)
    selection_reason: Optional[str] = None
    
@dataclass
class ExecutionResponse:
    run: Optional[RunResult]
    runs: Optional[Dict[str, RunResult]]
    evaluations: Optional[Dict[str, EvaluationResult]]
    comparison: Optional[ComparisonResult]


@dataclass
class LeaderboardEntry:
    model: str
    run: RunResult
    evaluation: EvaluationResult
    scores_by_strategy: Dict[str, float]
    ranks_by_strategy: Dict[str, int]
    narrative: str
    trend: Optional[Dict[str, Any]] = None
    latest_score: Optional[float] = None
    sample_count: int = 0


@dataclass
class LeaderboardResponse:
    mode: str
    sort_strategy: str
    page: int
    page_size: int
    total_items: int
    has_more: bool
    next_page: Optional[int]
    items: List[LeaderboardEntry]
    strategy_rankings: Dict[str, ComparisonResult]