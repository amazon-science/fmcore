from typing import Dict

from bears import FileMetadata

from fmcore.types.enums.dataset_enums import DatasetType
from fmcore.types.enums.prompt_tuner_enums import PromptTunerTaskType
from fmcore.types.typed import MutableTyped
from fmcore.types.prompt_tuner_types import (
    PromptTunerConfig,
)


class DatasetConfig(MutableTyped):
    inputs: Dict[DatasetType, FileMetadata] = {}
    output: FileMetadata


class BaseRunConfig(MutableTyped):
    dataset_config: DatasetConfig


class PromptTunerRunConfig(BaseRunConfig):
    task_type: PromptTunerTaskType
    prompt_tuner_config: PromptTunerConfig
