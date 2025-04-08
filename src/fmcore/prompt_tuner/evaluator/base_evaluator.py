from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from bears.util import Registry

from fmcore.prompt_tuner.evaluator.types.evaluator_types import EvaluatorConfig
from fmcore.types.typed import MutableTyped

I = TypeVar("I")  # Input Type
O = TypeVar("O")  # Output Type


class BaseEvaluator(Generic[I, O], MutableTyped, Registry, ABC):
    """
    Base class for all evaluators.

    This class defines the core evaluation interface and provides a registry-based
    mechanism for dynamically managing evaluator subclasses.

    Note: This is a throw away interface which will be replaced by Pipeline interface
        that would use transformers internally chaining operations one after the other

    Attributes:
        config (EvaluatorConfig): The configuration settings for the evaluator.
    """

    config: EvaluatorConfig

    @classmethod
    @abstractmethod
    def _get_instance(cls, *, evaluator_config: EvaluatorConfig) -> dict:
        """
        Returns an instance of the evaluator subclass, initialized using the given `evaluator_config`.

        This method must be implemented by each subclass to construct and return an instance
        of itself. It enables dynamic instantiation of LLM implementations while keeping the
        base class and registry mechanism unchanged.

        Args:
            evaluator_config (EvaluatorConfig): The configuration object containing settings for the Evaluator.

        Returns:
            BaseLLM: An instance of the subclass that extends `BaseLLM`.
        """
        pass

    @classmethod
    def of(cls, evaluator_config: EvaluatorConfig):
        """
        Factory method to instantiate the appropriate evaluator subclass based on evaluator_type.

        Args:
            evaluator_config (EvaluatorConfig): Configuration containing evaluator type and parameters.

        Returns:
            BaseEvaluator: An instance of the appropriate evaluator subclass.
        """
        BaseEvaluatorClass = BaseEvaluator.get_subclass(key=evaluator_config.evaluator_type)
        return BaseEvaluatorClass._get_instance(evaluator_config=evaluator_config)

    @abstractmethod
    def evaluate(self, data: I) -> O:
        """
        Synchronous evaluation method to process the input data and return an output.

        Args:
            data (I): The input data for evaluation.

        Returns:
            O: The output result of the evaluation.
        """
        pass

    @abstractmethod
    async def aevaluate(self, data: I) -> O:
        """
        Asynchronous evaluation method to process the input data and return an output.

        Args:
            data (I): The input data for evaluation.

        Returns:
            O: The output result of the evaluation.
        """
        pass
