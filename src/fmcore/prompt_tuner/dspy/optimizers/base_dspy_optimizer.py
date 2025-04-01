import dspy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, List

from fmcore.prompt_tuner.dspy.datasets.base_dataset import DspyDataset
from fmcore.prompt_tuner.types.optimizer_types import BaseOptimizerConfig
from fmcore.prompt_tuner.types.prompt_tuner_types import PromptTunerConfig
from fmcore.types.typed import MutableTyped
from bears.util import Registry


class BaseDspyOptimizer(MutableTyped, Registry, ABC):
    """
    Abstract base class for DSPy optimizers, providing common functionality
    for optimizing prompts using DSPy framework.

    Subclasses of this class are responsible for implementing the specific
    optimizers like BootStrap, MIProV2, COPRO etc.,
    https://dspy.ai/learn/optimization/optimizers/

    Attributes:
        module: The DSPy module used in the optimization process.
        evaluate: A callable function used to evaluate the performance of the model.
        optimizer_config: The configuration that defines the optimization strategy.
    """

    module: dspy.Module
    evaluate: Callable
    optimizer_config: BaseOptimizerConfig

    @classmethod
    @abstractmethod
    def _get_constructor_parameters(cls, *, prompt_tuner_config: PromptTunerConfig) -> Dict:
        """
        Abstract method to get the constructor parameters for the optimizer.

        Subclasses should implement this method to return a dictionary of
        parameters that are required to instantiate the optimizer.

        Args:
            prompt_tuner_config (PromptTunerConfig): Configuration for the prompt tuner.

        Returns:
            Dict: A dictionary of parameters required to construct the optimizer.
        """
        pass

    @classmethod
    def of(cls, prompt_tuner_config: PromptTunerConfig) -> "BaseDspyOptimizer":
        """
        Factory method to create an instance of a subclass of BaseDspyOptimizer
        using the provided configuration.

        Args:
            prompt_tuner_config (PromptTunerConfig): The configuration for the prompt tuner
                containing the optimizer configuration.

        Returns:
            BaseDspyOptimizer: An instance of the appropriate optimizer subclass.
        """
        BaseDspyOptimizerClass = BaseDspyOptimizer.get_subclass(
            key=prompt_tuner_config.optimizer_config.optimizer_type.name
        )
        constructor_parameters = BaseDspyOptimizerClass._get_constructor_parameters(
            prompt_tuner_config=prompt_tuner_config
        )
        return BaseDspyOptimizerClass(**constructor_parameters)

    @abstractmethod
    def optimize(self, dataset: DspyDataset) -> List[dspy.Module]:
        """
        Abstract method to perform the optimization on the provided dataset.

        Subclasses should implement this method to define the specific
        optimization process.

        Args:
            dataset (DspyDataset): The dataset containing the training and evaluation data.

        Returns:
            List[dspy.Module]: A list of optimized DSPy modules.
        """
        pass
