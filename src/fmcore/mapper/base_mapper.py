from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from fmcore.types.typed import MutableTyped

I = TypeVar("I")  # Input Type
O = TypeVar("O")  # Output Type


class BaseMapper(Generic[I, O], MutableTyped, ABC):
    """
    A generic base class for implementing mappers that process input data
    of type `I` and produce output of type `O`.

    This class enforces synchronous and asynchronous mapping methods
    through abstract methods `map` and `amap`.

    Type Parameters:
        I: The input data type.
        O: The output data type.
    """

    @abstractmethod
    def map(self, data: I) -> O:
        """
        Synchronously maps the input data into the desired output format.

        Args:
            data (I): The input data to map.

        Returns:
            O: The mapped output.
        """
        pass

    @abstractmethod
    async def amap(self, data: I) -> O:
        """
        Asynchronously maps the input data into the desired output format.

        Args:
            data (I): The input data to map.

        Returns:
            O: The mapped output.
        """
        pass
