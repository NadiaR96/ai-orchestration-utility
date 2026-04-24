from abc import ABC, abstractmethod


def output_length_penalty(metrics: dict) -> float:
    if "output_token_count" not in metrics:
        return 0.0

    token_count = float(metrics.get("output_token_count", 0.0) or 0.0)

    if token_count <= 0.0:
        return 0.4
    if token_count < 5.0:
        return 0.2
    return 0.0


class BaseScorer(ABC):
    @abstractmethod
    def compute(self, metrics: dict) -> float:
        pass