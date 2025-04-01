import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict

from fmcore.prompt_tuner.types.prompt_tuner_types import PromptTunerConfig, PromptTunerResult
from fmcore.types.enums.dataset_enums import DatasetType
from fmcore.types.typed import MutableTyped
from bears.util import Registry


class BasePromptTuner(MutableTyped, Registry, ABC):
    """
    Abstract base class for a prompt tuner. This class provides the
    necessary structure for creating specific prompt tuners based on
    the framework and configuration provided.

    Attributes:
        config (PromptTunerConfig): Configuration for the prompt tuner.

    TODO: This interface should be removed in the future by 01/04/2025
    """

    config: PromptTunerConfig

    @classmethod
    def of(cls, config: PromptTunerConfig) -> "BasePromptTuner":
        """
        Factory method to instantiate a specific prompt tuner based on the
        provided configuration.

        Args:
            config (PromptTunerConfig): Configuration used to instantiate the
            appropriate prompt tuner.

        Returns:
            BasePromptTuner: An instance of the correct subclass of BasePromptTuner
            based on the configuration's framework.
        """
        BasePromptTunerClass = BasePromptTuner.get_subclass(key=config.framework.name)
        return BasePromptTunerClass(config=config)

    @abstractmethod
    def tune(self, *, data: Dict[DatasetType, pd.DataFrame]) -> PromptTunerResult:
        """
        Abstract method to tune the prompts based on the provided dataset.

        Args:
            data (Dict[DatasetType, pd.DataFrame]): A dictionary mapping dataset types
            to pandas DataFrames containing the data to be used for tuning.

        Returns:
            PromptTunerResult: The result of the tuning process.
        """
        pass
