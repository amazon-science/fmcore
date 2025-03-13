from enum import auto
from autoenum import AutoEnum


class PromptTunerFramework(AutoEnum):
    LMOPS = auto()
    DSPY = auto()


class OptimizerType(AutoEnum):
    pass


class LMOpsOptimizerType(OptimizerType):
    pass


class DspyOptimizerType(OptimizerType):
    MIPRO_V2 = auto()
    COPRO = auto()
    BOOTSTRAP = auto()
