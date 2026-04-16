from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ExperimentConfig:
    name: str
    inputs: List[str]
    models: List[str]
    strategy: str = "balanced"
    runs_per_input: int = 1


@dataclass
class ExperimentResult:
    name: str
    comparisons: List[Dict[str, Any]]
    run_matrix: Dict[str, Any]
    summary: Dict[str, Any]