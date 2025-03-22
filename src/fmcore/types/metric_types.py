from abc import ABC
from typing import Dict, Optional
from pydantic import Field

from fmcore.types.typed import MutableTyped
from fmcore.types.enums.metric_enums import (
    EvaluationFieldType,
    MetricFramework,
    SupportedMetrics,
)
from fmcore.types.llm_types import LLMConfig


class MetricConfig(MutableTyped):
    """
    Configuration for metric evaluation.

    Attributes:
        metric_name: The type of metric to use
        framework: The framework to use for the metric
        metric_params: Parameters specific to the metric implementation
        llm_config: Configuration for the LLM used by the metric
        field_mapping: Maps from Labs field names to Customer dataset columns
                      e.g. {"INPUT": "question", "RESPONSE": "actual_output"}
    """

    metric_name: SupportedMetrics
    framework: Optional[MetricFramework] = None
    metric_params: Optional[Dict] = Field(default_factory=dict)
    llm_config: Optional[LLMConfig] = None
    field_mapping: Optional[Dict[EvaluationFieldType, str]] = Field(default_factory=dict)


class MetricResult(MutableTyped, ABC):
    pass


class ClassificationMetricResult(MetricResult):
    label: str
    confidence: Optional[float] = None


class CustomMetricResult(MetricResult):
    class Config(MetricConfig.Config):
        extra = "allow"


class TextGenerationMetricResult(MetricResult):
    score: float = 0.0
    reason: Optional[str] = None
