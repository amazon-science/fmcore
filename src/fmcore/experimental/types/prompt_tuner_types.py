from typing import Any, Dict, List, Optional, Union
from fmcore.experimental.types.enums.prompt_tuner_enums import (
    DspyOptimizerType,
    LMOpsOptimizerType,
    OptimizerType,
    PromptTunerFramework,
)
from fmcore.experimental.types.llm_types import LLMConfig
from fmcore.experimental.types.metric_types import MetricConfig
from fmcore.experimental.types.typed import MutableTyped


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
    params: Dict[str, Any]


class PromptTunerConfig(MutableTyped):
    framework: PromptTunerFramework
    prompt_config: PromptConfig
    optimzer_config: OptimizerConfig
