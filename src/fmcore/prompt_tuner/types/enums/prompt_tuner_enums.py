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