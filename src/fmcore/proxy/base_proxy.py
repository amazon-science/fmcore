from abc import ABC
from typing import Generic, TypeVar
from bears.util import MutableParameters, Registry

T = TypeVar("T")  # Represents any client type


class BaseProxy(MutableParameters, Registry, ABC, Generic[T]):
    """Base class for all proxy implementations, supporting any client type."""

    client: T  # Generic client instance
