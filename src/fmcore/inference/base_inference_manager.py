from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from bears.util import Registry

from fmcore.inference.types.inference_manager_types import InferenceManagerConfig
from fmcore.types.typed import MutableTyped

I = TypeVar("I")  # Input Type
O = TypeVar("O")  # Output Type


class BaseInferenceManager(MutableTyped, Generic[I, O], Registry, ABC):
    """
    Abstract base class for implementing inference managers.

    This class defines the interface and lifecycle for inference manager implementations,
    supporting configurable initialization, dynamic subclass resolution via a registry,
    and type-safe execution of inference on input datasets.

    Type Parameters:
        I: Input dataset type.
        O: Output result type.

    Attributes:
        config (InferenceManagerConfig): Configuration for the inference manager instance.
    """

    config: InferenceManagerConfig

    @classmethod
    @abstractmethod
    def _get_instance(cls, *, config: InferenceManagerConfig) -> "BaseInferenceManager":
        """
        Returns an instance of the inference manager configured with the given parameters.

        Args:
            config (InferenceManagerConfig): Configuration used to initialize the manager.

        Returns:
            BaseInferenceManager: A configured inference manager instance.
        """
        pass

    @classmethod
    def of(cls, config: InferenceManagerConfig):
        """
        Factory method to construct an inference manager based on the given config.

        Resolves the appropriate subclass from the registry using the `inference_manager_type`
        and delegates instantiation to the subclass.

        Args:
            config (InferenceManagerConfig): Configuration that determines the subclass.

        Returns:
            BaseInferenceManager: An instance of the resolved subclass.
        """
        BaseInferenceManagerClass = BaseInferenceManager.get_subclass(key=config.inference_manager_type)
        return BaseInferenceManagerClass._get_instance(config=config)

    @abstractmethod
    def run(self, dataset: I) -> O:
        """
        Executes inference on the given dataset.

        Args:
            dataset (I): Input dataset for inference.

        Returns:
            O: Inference result.
        """
        pass
