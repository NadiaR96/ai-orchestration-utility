from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union


DecisionState = Literal[
    "RECOMMENDED",
    "CONSTRAINED",
    "ABSTAIN",
    "INVALID",
]

SystemHealth = Literal[
    "OK",
    "DEGRADED",
    "FAIL",
]

EvaluationStatus = Literal[
    "VALID",
    "WEAK_SIGNAL",
    "INVALID",
    "INSUFFICIENT_DATA",
    "NOISY",
    "UNSTABLE",
]

FailureMode = Literal[
    "LOW_QUALITY",
    "LOW_SEPARATION",
    "INSUFFICIENT_DATA",
    "HIGH_VARIANCE",
    "COST_DOMINATED",
    "LATENCY_DOMINATED",
    "ALL_MODELS_WEAK",
    "LOW_CONSISTENCY",
    "USE_CASE_MISMATCH",
    "SCORE_SCALE_DRIFT",
]


class CompareRequest(BaseModel):
    input: str = Field(..., min_length=1)
    models: List[str] = Field(default_factory=lambda: ["small"])
    strategy: str = Field(default="balanced")
    retrieval: Literal["rag", "none"] = Field(default="rag")
    use_case: Optional[str] = Field(default=None, description="Optional use-case tag for recommendation matching")
    reference: Optional[Union[str, List[str]]] = Field(default=None, description="Optional ground-truth reference(s) for quality metric computation")


class RunRequest(BaseModel):
    input: str = Field(..., min_length=1)
    model: str = Field(default="small")
    strategy: str = Field(default="balanced")
    retrieval: Literal["rag", "none"] = Field(default="rag")
    use_case: Optional[str] = Field(default=None, description="Optional use-case tag for recommendation matching")
    reference: Optional[Union[str, List[str]]] = Field(default=None, description="Optional ground-truth reference(s) for quality metric computation")


class LeaderboardPromptRequest(BaseModel):
    input: str = Field(..., min_length=1)
    reference: Optional[str] = None
    models: List[str] = Field(default_factory=lambda: ["small", "default", "quality"])
    retrieval: str = Field(default="rag")
    sort_strategy: str = Field(default="balanced")
    aggregation: Literal["latest"] = "latest"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)


class LeaderboardHistoryQuery(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)
    sort_strategy: str = Field(default="balanced")
    aggregation: Literal["latest"] = "latest"
    models: Optional[List[str]] = None


class RecommendationRequest(BaseModel):
    use_case: str = Field(..., min_length=1, description="Task type or domain (e.g. 'summarisation', 'code generation')")
    strategy: str = Field(default="balanced", description="Scoring strategy to rank by")
    top_n: int = Field(default=3, ge=1, le=20, description="Maximum number of alternatives to return")
    min_samples: int = Field(default=1, ge=1, description="Minimum number of logged runs a model must have to qualify")
    source: Literal["live", "experiment", "all"] = Field(default="all", description="Log source scope to query")


class AlternativeModel(BaseModel):
    model: str
    score: float
    sample_count: int
    avg_latency: float
    avg_cost: float
    score_stddev: float
    score_delta_from_best: float
    confidence: float
    p95_latency: float
    consistency_above_threshold: float
    avg_total_tokens: float = 0.0
    avg_cost_per_1k_tokens: float = 0.0
    avg_quality_per_1k_tokens: float = 0.0


class ScoreVector(BaseModel):
    quality_score: float = Field(..., description="Primary model quality score")
    latency_s: float
    cost_usd: float
    variance: float
    sample_size: int


class EvaluationMetrics(BaseModel):
    score_margin: float
    score_variance: float
    sample_count: int
    p95_latency_s: Optional[float] = None
    consistency_above_threshold: Optional[float] = None


class Decision(BaseModel):
    state: DecisionState
    selected_model: Optional[str] = None
    reason: Optional[str] = None


class DecisionReliability(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    label: Literal["LOW", "MEDIUM", "HIGH", "INSUFFICIENT EVIDENCE"]


class FailureAnalysis(BaseModel):
    modes: List[FailureMode] = Field(default_factory=list)
    primary_cause: Optional[str] = None


class ModelAlternative(BaseModel):
    model: str
    score: float
    delta_from_best: float
    avg_latency_s: float
    avg_cost_usd: float
    confidence: float


class EvaluationResponse(BaseModel):
    system_health: SystemHealth
    evaluation_status: EvaluationStatus
    decision: Decision
    reliability: DecisionReliability
    metrics: EvaluationMetrics
    failure_analysis: FailureAnalysis
    alternatives: List[ModelAlternative]
    score_vector: ScoreVector
    strategy: str
    use_case: str
    use_case_matched: bool


class RecommendationResponse(BaseModel):
    best_model: str
    best_score: float
    recommendation_available: bool
    no_recommendation_reason: Optional[str] = None
    gate_status: str
    gate_threshold: float
    gate_triggers: List[str]
    validity_status: str
    is_valid: bool
    validity_reasons: List[str]
    confidence_reasons: List[str]
    alternatives: List[AlternativeModel]
    justification: str
    evaluation: EvaluationResponse