from typing import Any, Dict, List, Optional, Union

import pandas as pd

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


class OptimizerConfig(MutableTyped):
    type: Union[LMOpsOptimizerType, DspyOptimizerType]
    student_config: LLMConfig
    teacher_config: Optional[LLMConfig]
    metric_config: MetricConfig
    optimizer_params: Dict[str, Any] = {}  # TODO: Think Again


class PromptTunerConfig(MutableTyped):
    framework: PromptTunerFramework
    prompt_config: PromptConfig
    optimzer_config: OptimizerConfig


class EvaluationResult(MutableTyped):
    score: float
    result: Optional[pd.DataFrame]

class OptimizedPrompt(MutableTyped):
    template: str
    validation_result: Optional[EvaluationResult]
    test_result: Optional[EvaluationResult]


class PromptTunerResult(MutableTyped):
    prompts: List[OptimizedPrompt]
