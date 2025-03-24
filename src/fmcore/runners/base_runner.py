from abc import ABC, abstractmethod
from typing import NoReturn
from fmcore.types.typed import MutableTyped
from bears.util import Registry


class BaseRunner(MutableTyped, Registry, ABC):

    @abstractmethod
    def run(self, run_config: dict) -> NoReturn:
        pass
