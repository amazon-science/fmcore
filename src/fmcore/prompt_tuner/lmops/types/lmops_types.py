from typing import Optional, List

from pydantic import Field

from fmcore.prompt_tuner.types.enums.optimizer_enums import OptimizerMetricType, LMOPSOptimizerType
from fmcore.prompt_tuner.types.mixins.optimizer_mixins import (
    StudentConfigMixin,
    TeacherConfigMixin,
    EvaluatorConfigMixin,
)
from fmcore.prompt_tuner.types.optimizer_types import BaseOptimizerParams, BaseOptimizerConfig


class LMOPSClassificationOptimizerParams(BaseOptimizerParams):
    """
    Parameters for the LMOPS Classification Optimizer.

    Inherits from `BaseOptimizerParams` and sets the default optimizer metric to `ACCURACY`.
    """

    optimizer_metric: OptimizerMetricType = OptimizerMetricType.ACCURACY
    optimizer_section_tag: str = Field(default="")
    rounds: Optional[int] = Field(default=1)
    num_candidates_to_generate: Optional[int] = Field(default=4)
    categories: List[str]
    budget: str = Field(default="$ value")


class LMOPSClassificationOptimizerConfig(
    BaseOptimizerConfig, StudentConfigMixin, TeacherConfigMixin, EvaluatorConfigMixin
):
    """
    Configuration for the LMOPS Classification Optimizer.

    Inherits from `BaseOptimizerConfig` and defines various optimizer types, as well as optional
    parameters specific to the LMOPS classification optimization.

    Attributes:
        aliases (list): A list of optimizer types related to classification tasks.
        optimizer_params (Optional[LMOPSClassificationOptimizerParams]): Optional parameters
        specific to the LMOPS classification optimizer, such as optimizer metrics.
    """

    aliases = [
        LMOPSOptimizerType.BINARY_CLASSIFICATION,
        LMOPSOptimizerType.MULTI_CLASS_CLASSIFICATION_COMBINED,
        LMOPSOptimizerType.MULTI_CLASS_CLASSIFICATION_GRADIENT,
        LMOPSOptimizerType.MULTI_CLASS_CLASSIFICATION_PAIRWISE,
        LMOPSOptimizerType.MULTI_CLASS_CLASSIFICATION_ONE_VS_REST,
        LMOPSOptimizerType.MULTI_CLASS_CLASSIFICATION_INSTRUCTION_SCORING,
    ]
    optimizer_params: Optional[LMOPSClassificationOptimizerParams]
