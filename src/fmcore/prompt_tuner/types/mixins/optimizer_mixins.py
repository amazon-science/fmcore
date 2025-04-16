from typing import Optional, Union

from pydantic import Field

from fmcore.llm.types.llm_types import LLMConfig, DistributedLLMConfig
from fmcore.prompt_tuner.evaluator.types.evaluator_types import EvaluatorConfig
from fmcore.types.mixins_types import Mixin
from fmcore.types.typed import MutableTyped


class StudentConfigMixin(MutableTyped, Mixin):
    """
    Mixin for Student LLM configuration.

    Attributes:
        student_config (Optional[LLMConfig]): The LLM configuration object for student model
    """

    student_config: Union[LLMConfig, DistributedLLMConfig] = Field(union_mode="left_to_right")


class TeacherConfigMixin(MutableTyped, Mixin):
    """
    Mixin for Student LLM configuration.

    Attributes:
        teacher_config (Optional[LLMConfig]): The LLM configuration object for teacher model
    """

    teacher_config: Union[LLMConfig, DistributedLLMConfig] = Field(union_mode="left_to_right")


class EvaluatorConfigMixin(MutableTyped, Mixin):
    """
    Mixin for Evaluator Config configuration.

    Attributes:
        evaluator_config (Optional[EvaluatorConfig]): The LLM configuration object for evaluator model
    """

    evaluator_config: EvaluatorConfig
