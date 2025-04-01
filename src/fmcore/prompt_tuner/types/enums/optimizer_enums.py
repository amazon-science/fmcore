from enum import Enum


class OptimizerType(str, Enum):
    """
    Base Enum class for defining different types of optimizers.

    This class is intended to be extended by other enums to represent
    specific categories of optimizers.
    """

    pass


class DSPyOptimizerType(OptimizerType):
    """
    Enum class representing specific optimizer types for DSPY.

    This class extends `OptimizerType` and defines optimizers that are
    specific to the DSPY framework.
    """

    MIPRO_V2 = "MIPRO_V2"


class LMOPSOptimizerType(OptimizerType):
    """
    Enum class representing optimizer types for LMOps.

    This class extends `OptimizerType` and defines a variety of optimizer
    configurations used in the LMOps library, specifically for multi-class
    classification tasks.
    """

    BINARY_CLASSIFICATION = "BINARY_CLASSIFICATION"
    MULTI_CLASS_CLASSIFICATION_COMBINED = "MULTI_CLASS_CLASSIFICATION_COMBINED"
    MULTI_CLASS_CLASSIFICATION_GRADIENT = "MULTI_CLASS_CLASSIFICATION_GRADIENT"
    MULTI_CLASS_CLASSIFICATION_PAIRWISE = "MULTI_CLASS_CLASSIFICATION_PAIRWISE"
    MULTI_CLASS_CLASSIFICATION_ONE_VS_REST = "MULTI_CLASS_CLASSIFICATION_ONE_VS_REST"
    MULTI_CLASS_CLASSIFICATION_INSTRUCTION_SCORING = "MULTI_CLASS_CLASSIFICATION_INSTRUCTION_SCORING"


class OptimizerMetricType(str, Enum):
    """
    Enum class representing different metric types used for optimizer evaluation.

    This class defines the available metrics to be used for evaluating the
    performance of an optimizer.
    """

    ACCURACY = "ACCURACY"
