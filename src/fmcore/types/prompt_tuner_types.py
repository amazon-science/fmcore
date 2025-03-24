from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import Extra, Field

from fmcore.types.typed import MutableTyped
from fmcore.types.enums.prompt_tuner_enums import (
    DspyOptimizerType,
    LMOpsOptimizerType,
    PromptTunerFramework,
)
from fmcore.types.llm_types import LLMConfig
from fmcore.types.metric_types import MetricConfig


class PromptField(MutableTyped):
    name: str
    description: str


class PromptConfig(MutableTyped):
    prompt: str
    input_fields: List[PromptField]
    output_fields: List[PromptField]


class BaseOptimizerTypeConfig(MutableTyped):
    student_config: LLMConfig
    teacher_config: Optional[LLMConfig]
    metric_config: MetricConfig


class MIPROV2OptimizerType(BaseOptimizerTypeConfig):
    type: DspyOptimizerType = DspyOptimizerType.MIPRO_V2
    metric_config: MetricConfig
    optimizer_params: Dict[str, Any] = {}  # TODO: Think Again


class PromptTunerConfig(MutableTyped):
    framework: PromptTunerFramework
    prompt_config: PromptConfig
    optimizer_config: Union[MIPROV2OptimizerType]


class PromptEvaluationResult(MutableTyped):
    score: float
    data: Optional[pd.DataFrame]


class TunedPrompt(MutableTyped):
    prompt_id: str
    prompt: str
    validation_result: Optional[PromptEvaluationResult]
    test_result: Optional[PromptEvaluationResult]


class PromptTunerResult(MutableTyped):
    prompts: List[TunedPrompt]
