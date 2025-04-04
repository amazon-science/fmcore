import dspy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, List

from fmcore.prompt_tuner.dspy.datasets.base_dataset import DspyDataset
from fmcore.prompt_tuner.types.optimizer_types import BaseOptimizerConfig
from fmcore.prompt_tuner.types.prompt_tuner_types import PromptTunerConfig
from fmcore.types.typed import MutableTyped
from bears.util import Registry


class BaseDspyOptimizerWrapper(MutableTyped, Registry, ABC):
    """
    DO NOT WRITE OPTIMIZERS INHERITING THIS INTERFACE

    Abstract base class for DSPy optimizers, providing common functionality
    for optimizing prompts using the DSPy framework.

    This is a wrapper over DSPy optimizers because DSPy does not provide a unified interface
    for different optimizers. The inputs and outputs of optimizers like BootStrap, MIProV2,
    and COPRO vary significantly, requiring specific transformation logic. This class
    ensures that the necessary parsing and transformation are handled at the optimizer level,
    without implementing optimizers directly.

    Subclasses of this class should implement optimizer-specific logic.
    Reference: https://dspy.ai/learn/optimization/optimizers/

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
    def _get_instance(cls, *, prompt_tuner_config: PromptTunerConfig) -> "BaseDspyOptimizerWrapper":
        """
        Creates and returns an instance of a subclass implementing BaseDspyOptimizerWrapper.

        Subclasses must implement this method to instantiate and return an optimizer
        based on the given prompt tuner configuration.

        Args:
            prompt_tuner_config (PromptTunerConfig): The configuration for the prompt tuner.

        Returns:
            BaseDspyOptimizerWrapper: An instance of the optimizer.
        """
        pass

    @classmethod
    def of(cls, prompt_tuner_config: PromptTunerConfig) -> "BaseDspyOptimizerWrapper":
        """
        Factory method to create an instance of a subclass of BaseDspyOptimizer
        using the provided configuration.

        Args:
            prompt_tuner_config (PromptTunerConfig): The configuration for the prompt tuner
                containing the optimizer configuration.

        Returns:
            BaseDspyOptimizerWrapper: An instance of the appropriate optimizer subclass.
        """
        BaseDspyOptimizerClass = BaseDspyOptimizerWrapper.get_subclass(
            key=prompt_tuner_config.optimizer_config.optimizer_type
        )
        return BaseDspyOptimizerClass._get_instance(prompt_tuner_config=prompt_tuner_config)

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
