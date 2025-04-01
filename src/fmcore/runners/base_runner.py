from abc import ABC, abstractmethod
from typing import NoReturn
from fmcore.types.typed import MutableTyped
from bears.util import Registry


class BaseRunner(MutableTyped, Registry, ABC):
    """
    Abstract base class for all runners.

    This class provides a common interface for executing different types of runs.

    Methods:
        run(run_config: dict) -> NoReturn:
            Abstract method that must be implemented by subclasses to execute a run.
    """

    @abstractmethod
    def run(self, run_config: dict) -> NoReturn:
        """Execute a run based on the given configuration."""
        pass
