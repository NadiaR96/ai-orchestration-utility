from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def run(self, prompt: str) -> str:
        pass