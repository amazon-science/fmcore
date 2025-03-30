from enum import Enum


class EvaluatorType(str, Enum):
    """
    Enum class representing different types of evaluators.
    """
    BOOLEAN_LLM_JUDGE = "BOOLEAN_LLM_JUDGE"