from enum import Enum


class EvaluatorType(str, Enum):
    """
    Enum class representing different types of evaluators.
    """

    LLM_AS_A_JUDGE_BOOLEAN = "LLM_AS_A_JUDGE_BOOLEAN"
    CLASSIFICATION = "CLASSIFICATION"
