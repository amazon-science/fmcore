from enum import Enum


class PromptTunerFramework(str, Enum):
    """
    Enum representing supported prompt tuner frameworks.

    Attributes:
        DSPY: Represents the DSPy framework.
        LMOPS: Represents the LMOps framework.
    """

    DSPY = "DSPY"
    LMOPS = "LMOPS"


class PromptTunerTaskType(str, Enum):
    """
    Enum representing different types of prompt tuning tasks.

    Attributes:
        BINARY_CLASSIFICATION: Task for binary classification problems.
        MULTI_CLASS_CLASSIFICATION: Task for multi-class classification problems.
        TEXT_GENERATION: Task for text generation problems.
    """

    BINARY_CLASSIFICATION = "BINARY_CLASSIFICATION"
    MULTI_CLASS_CLASSIFICATION = "MULTI_CLASS_CLASSIFICATION"
    TEXT_GENERATION = "TEXT_GENERATION"
