from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from bears.util import Registry

from fmcore.evaluator.types.evaluator_types import EvaluatorConfig
from fmcore.types.typed import MutableTyped

I = TypeVar("I")  # Input Type
O = TypeVar("O")  # Output Type


class BaseEvaluator(Generic[I, O], MutableTyped, Registry, ABC):
    """
    Base class for all evaluators.

    This class defines the core evaluation interface and provides a registry-based
    mechanism for dynamically managing evaluator subclasses.

    Attributes:
        config (EvaluatorConfig): The configuration settings for the evaluator.
    """

    config: EvaluatorConfig

    @classmethod
    @abstractmethod
    def _get_constructor_parameters(cls, *, evaluator_config: EvaluatorConfig) -> dict:
        """
        Generate the constructor parameters required for initializing a subclass.

        This method is intended to be implemented by each subclass to provide the necessary parameters dynamically
        based on the given `llm_config`.The purpose of this abstraction is to ensure that the registry pattern used
        for creating subclasses does not require modification when new subclasses are introduced.

        Each subclass has a different set of constructor parameters, and this method allows them to generate those
        parameters as needed, ensuring flexibility and maintainability in the codebase.

        ---

        **How This Aligns with the Open/Closed Principle (OCP)**
        The **Open/Closed Principle** states that a system should be **open for extension but closed for modification**.
         This method enforces OCP by allowing new subclasses to be introduced **without modifying the base class** or
         the registry/factory mechanism. Instead of altering existing code when a new subclass is introduced, the new
         subclass simply implements `get_constructor_paramters`,ensuring it provides the appropriate constructor arguments.

        **Closed for Modification**:
        - The registry/factory does not need to be modified when adding a new subclass.
        - The base class remains unchanged, preventing regressions in existing functionality.

        **Open for Extension**:
        - New LLM subclasses can be introduced freely, each defining how to extract its own constructor parameters.
        - Different LLM implementations can have varying configurations, yet still be instantiated seamlessly within the existing system.

        ---

        Args:
            evaluator_config (EvaluatorConfig): The configuration object containing Evaluator-related settings.

        Returns:
            dict: A dictionary of keyword arguments (`**kwargs`) that can be used to instantiate the subclass dynamically.

        By using this approach, we ensure that `BaseLLM` remains **stable and unmodified**, while allowing new
        subclasses to introduce their own custom constructor parameters without breaking existing functionality.
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
        BaseEvaluatorClass = BaseEvaluator.get_subclass(key=evaluator_config.evaluator_type.name)
        constructor_params = BaseEvaluatorClass._get_constructor_parameters(evaluator_config=evaluator_config)
        return BaseEvaluatorClass(**constructor_params)

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
