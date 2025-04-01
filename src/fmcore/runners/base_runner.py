from abc import ABC, abstractmethod
from typing import NoReturn, TypeVar, Generic
from fmcore.types.typed import MutableTyped
from bears.util import Registry

I = TypeVar("I")  # Input Type


class BaseRunner(Generic[I], MutableTyped, Registry, ABC):
    """
    Abstract base class for all runners.

    This class provides a common interface for executing different types of runs.
    It is intended to be subclassed and customized for specific types of runs.

    Subclasses must implement the `run` method to define how to execute the run.

    Methods:
        run(run_config: I) -> NoReturn:
            Abstract method that must be implemented by subclasses to execute a run.
            It takes in a configuration and performs the necessary steps to run the process.
    """

    @abstractmethod
    def run(self, run_config: I) -> NoReturn:
        """
        Execute a run based on the given configuration.

        Args:
            run_config (I): The configuration required to execute the run.

        Returns:
            NoReturn: This method does not return any value.
        """
        pass
