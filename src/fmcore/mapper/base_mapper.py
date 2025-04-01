from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from fmcore.types.typed import MutableTyped

I = TypeVar("I")  # Input Type
O = TypeVar("O")  # Output Type


# TODO Evaluate if this can be replaced with SKlearn Interfaces
class BaseMapper(MutableTyped, Generic[I, O], ABC):
    """
    A generic base class for implementing mappers that process input data
    of type `I` and produce output of type `O`.

    This class enforces synchronous and asynchronous mapping methods
    through abstract methods `map` and `amap`.

    It is conceptually similar to the `map` function in functional programming,
    where a transformation function is applied to each element of a collection.
    This paradigm is foundational in functional languages like Lisp and widely
    adopted in large-scale data processing frameworks such as MapReduce, Apache Hadoop,
    and Apache Spark for efficient parallel transformations.

    Additionally, `BaseMapper` serves a similar role to Java’s `Function<T, R>` interface,
    which defines a contract for transforming an input of type `T` into an output of type `R`.
    Just as `Function<T, R>` is used for functional composition in Java streams and reactive
    programming, `BaseMapper` provides a structured approach for transformation logic
    in a generic, reusable manner.

    Type Parameters:
        I: The input data type.
        O: The output data type.
    """

    @abstractmethod
    def map(self, data: I) -> O:
        """
        Synchronously maps the input data into the desired output format.

        This method follows the functional programming principle of applying
        a transformation function to an input, similar to `map` in Python, Java, and Scala.

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

        This method is useful for distributed or I/O-bound operations, similar to how
        map functions are used in parallel processing frameworks like Hadoop and Spark.

        Args:
            data (I): The input data to map.

        Returns:
            O: The mapped output.
        """
        pass
