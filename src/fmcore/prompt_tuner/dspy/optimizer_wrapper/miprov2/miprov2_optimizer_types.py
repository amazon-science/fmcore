from typing import Optional

from fmcore.prompt_tuner.types.enums.optimizer_enums import OptimizerMetricType, DSPyOptimizerType
from fmcore.prompt_tuner.types.mixins.optimizer_mixins import (
    StudentConfigMixin,
    TeacherConfigMixin,
    EvaluatorConfigMixin,
)
from fmcore.prompt_tuner.types.optimizer_types import BaseOptimizerParams, BaseOptimizerConfig


class MIPROv2OptimizerParams(BaseOptimizerParams):
    """
    Parameters for the MIPROv2 optimizer.

    This class contains configuration options specific to the MIPROv2 optimizer, including
    the objective metric to optimize and any auto-tuning options.

    Attributes:
        objective_metric (OptimizerMetricType): The metric to optimize for, defaults to ACCURACY.
        auto (Optional[str]): An optional auto-tuning setting, defaults to 'light'.
    """

    optimizer_metric: str = OptimizerMetricType.ACCURACY
    auto: Optional[str] = "light"
    num_candidates: int = 7


class MIPROv2OptimizerConfig(
    BaseOptimizerConfig, StudentConfigMixin, TeacherConfigMixin, EvaluatorConfigMixin
):
    """
    Configuration for the MIPROv2 optimizer.

    This class combines configuration options for the student model, teacher model, evaluator,
    and the MIPROv2 optimizer parameters.

    Attributes:
        aliases (list): List of supported optimizer types, includes MIPRO_V2.
        optimizer_params (Optional[MIPROv2OptimizerParams]): Configuration parameters for the optimizer.
    """

    aliases = [DSPyOptimizerType.MIPRO_V2]
    optimizer_params: Optional[MIPROv2OptimizerParams]
