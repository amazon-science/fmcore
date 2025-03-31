from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from ..types.typed import MutableTyped
from langchain_core.messages import BaseMessage

O = TypeVar("O")

class BasePredictor(Generic[O], MutableTyped, ABC):
    """Base class for all predictors that handle message inputs."""

    @abstractmethod
    def predict(self, data: List[BaseMessage]) -> O:
        """
        Predict based on input messages.

        Args:
            data: List of messages to process

        Returns:
            Prediction result of type O
        """
        pass

    @abstractmethod
    async def apredict(self, data: List[BaseMessage]) -> O:
        """
        Asynchronously predict based on input messages.

        Args:
            data: List of messages to process

        Returns:
            Prediction result of type O
        """
        pass