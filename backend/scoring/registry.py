from backend.scoring.quality import QualityScorer
from backend.scoring.cost_aware import CostAwareScorer
from backend.scoring.rag_aware import RAGScorer
from backend.scoring.balanced import BalancedScorer


SCORERS = {
    "quality": QualityScorer,
    "cost_aware": CostAwareScorer,
    "rag": RAGScorer,
    "balanced": BalancedScorer,
}


def get_scorer(name: str):
    name = name.lower()

    if name not in SCORERS:
        raise ValueError(f"Unknown scoring strategy: {name}")

    return SCORERS[name]()  # instantiate here