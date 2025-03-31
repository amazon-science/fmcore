from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from fmcore.types.typed import MutableTyped

I = TypeVar("I")  # Input Type
O = TypeVar("O")  # Output Type


class BaseTransformer(Generic[I, O], MutableTyped, ABC):
    """
    A generic base class for implementing transformers that process input data
    of type `I` and produce output of type `O`.

    This class enforces synchronous and asynchronous transformation methods
    through abstract methods `transform` and `atransform`.

    Type Parameters:
        I: The input data type.
        O: The output data type.
    """

    @abstractmethod
    def transform(self, data: I) -> O:
        """
        Synchronously transforms the input data into the desired output format.

        Args:
            data (I): The input data to transform.

        Returns:
            O: The transformed output.
        """
        pass

    @abstractmethod
    async def atransform(self, data: I) -> O:
        """
        Asynchronously transforms the input data into the desired output format.

        Args:
            data (I): The input data to transform.

        Returns:
            O: The transformed output.
        """
        pass
