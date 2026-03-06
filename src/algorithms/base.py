from abc import ABC, abstractmethod
import numpy as np

class BaseAlgorithm(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass