from abc import ABC, abstractmethod

class BaseScorer(ABC):
    @abstractmethod
    def compute(self, metrics: dict) -> float:
        pass