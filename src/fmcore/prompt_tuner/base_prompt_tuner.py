import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict
from fmcore.types.enums.dataset_enums import DatasetType
from fmcore.types.prompt_tuner_types import PromptTunerConfig, PromptTunerResult
from fmcore.types.typed import MutableTyped
from bears.util import Registry


class BasePromptTuner(MutableTyped, Registry, ABC):
    config: PromptTunerConfig

    @classmethod
    @abstractmethod
    def _get_constructor_parameters(cls, *, config: PromptTunerConfig) -> Dict:
        pass

    @classmethod
    def of(cls, config: PromptTunerConfig) -> "BasePromptTuner":
        BasePromptTunerClass = BasePromptTuner.get_subclass(key=config.framework)
        constructor_parameters = BasePromptTunerClass._get_constructor_parameters(config=config)
        return BasePromptTunerClass(**constructor_parameters)

    @abstractmethod
    def tune(self, *, data: Dict[DatasetType, pd.DataFrame]) -> PromptTunerResult:
        pass
