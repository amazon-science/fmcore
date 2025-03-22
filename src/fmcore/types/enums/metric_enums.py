from enum import auto
from autoenum import AutoEnum


class MetricFramework(AutoEnum):
    """Enum for different frameworks for metrics."""

    DEEPEVAL = auto()
    CUSTOM = auto()


class SupportedMetrics(AutoEnum):
    """Enum for supported metrics."""

    DEEPEVAL_GEVAL = auto()
    CUSTOM = auto()


class EvaluationFieldType(AutoEnum):
    """Standard field types for evaluation metrics across different frameworks."""

    INPUT = auto()
    CONTEXT = auto()
    GROUND_TRUTH = auto()
    OUTPUT = auto()
    RESPONSE = auto()
