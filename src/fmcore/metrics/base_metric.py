from abc import ABC, abstractmethod
from typing import Dict

from fmcore.types.metric_types import MetricConfig, MetricResult
from fmcore.types.typed import MutableTyped
from bears.util import Registry


class BaseMetric(MutableTyped, Registry, ABC):
    """
    Abstract base class for implementing evaluation metrics.

    Attributes:
        config (MetricConfig): Configuration settings for the metric implementation.
    """

    config: MetricConfig

    @classmethod
    @abstractmethod
    def _get_constructor_parameters(cls, *, metric_config: MetricConfig) -> dict:
        """
        Generate the constructor parameters required for initializing a subclass.

        This method is intended to be implemented by each subclass to provide the necessary parameters
        dynamically based on the given metric configuration. It serves as a crucial part of the factory
        pattern implementation, allowing flexible instantiation of different metric types.

        Args:
            metric_config (MetricConfig): Configuration object containing metric-specific settings
                and parameters needed for initialization.

        Returns:
            dict: A dictionary of keyword arguments that will be used to instantiate the subclass.
                The keys should match the parameter names in the subclass's __init__ method.

        """
        pass

    @classmethod
    def of(cls, metric_config: MetricConfig):
        """
        Factory method to create an instance of the appropriate Metric subclass.

        This method implements a hybrid pattern combining Registry and Factory patterns to
        dynamically instantiate the correct metric implementation based on the configuration.
        It first looks up the appropriate subclass using the metric name from the registry,
        then creates an instance with the necessary parameters.

        Args:
            metric_config (MetricConfig): Configuration object that specifies which metric
                to instantiate and its initialization parameters.

        Returns:
            BaseMetric: An initialized instance of the specified metric subclass.
        """

        BaseMetricClass = BaseMetric.get_subclass(key=metric_config.metric_name.name)
        constructor_params = BaseMetricClass._get_constructor_parameters(
            metric_config=metric_config
        )
        return BaseMetricClass(**constructor_params)

    @abstractmethod
    def evaluate(self, data: Dict) -> MetricResult:
        """
        Synchronously evaluate the metric on the provided input data.

        This method should be implemented by subclasses to perform the actual metric
        calculation on the input data according to the metric's specific logic.

        Args:
            data (Dict): Input data to evaluate. The structure of this dictionary
                should match the expected format for the specific metric implementation.

        Returns:
            MetricResult: The calculated metric result containing the score and any
                additional metadata specific to the metric implementation.
        """
        pass

    @abstractmethod
    async def aevaluate(self, data: Dict) -> MetricResult:
        """
        Asynchronously evaluate the metric on the provided input data.

        This method provides an asynchronous interface for metric evaluation, which
        is particularly useful for metrics that need to perform I/O operations or
        make external API calls during evaluation.

        Args:
            data (Dict): Input data to evaluate. The structure of this dictionary
                should match the expected format for the specific metric implementation.

        Returns:
            MetricResult: The calculated metric result containing the score and any
                additional metadata specific to the metric implementation.
        """
        pass
