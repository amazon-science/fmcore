from fmcore.prompt_tuner.types.enums.prompt_tuner_enums import PromptTunerTaskType
from fmcore.prompt_tuner.types.prompt_tuner_types import PromptTunerConfig
from fmcore.types.config_types import DatasetConfig
from fmcore.types.typed import MutableTyped


class PromptTunerRunConfig(MutableTyped):
    """
    Configuration for running a prompt tuning task.

    Attributes:
        task_type (PromptTunerTaskType): The type of prompt tuning task to be executed.
        prompt_tuner_config (PromptTunerConfig): Configuration for the prompt tuning process.
    """

    task_type: PromptTunerTaskType
    dataset_config: DatasetConfig
    prompt_tuner_config: PromptTunerConfig
