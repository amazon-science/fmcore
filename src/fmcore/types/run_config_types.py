from fmcore.types.typed import MutableTyped

from fmcore.types.dataset_types import DatasetConfig
from fmcore.types.prompt_tuner_types import (
    OptimizerConfig,
    PromptConfig,
    PromptTunerConfig,
)


class BaseRunConfig(MutableTyped):
    dataset_config: DatasetConfig

class PromptTunerRunConfig(BaseRunConfig):
    prompt_tuner_config: PromptTunerConfig
