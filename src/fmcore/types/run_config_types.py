from typing import Dict

from bears import FileMetadata

from fmcore.prompt_tuner.types.enums.prompt_tuner_enums import PromptTunerTaskType
from fmcore.prompt_tuner.types.prompt_tuner_types import PromptTunerConfig
from fmcore.types.enums.dataset_enums import DatasetType
from fmcore.types.typed import MutableTyped


class DatasetConfig(MutableTyped):
    """
    Configuration for dataset storage and file references.

    Attributes:
        inputs (Dict[DatasetType, FileMetadata]): Mapping of dataset types (TRAIN, TEST, VAL) to file metadata.
        output (FileMetadata): Metadata for the output file.
    """

    inputs: Dict[DatasetType, FileMetadata] = {}
    output: FileMetadata


class BaseRunConfig(MutableTyped):
    """
    Base configuration for a model run.

    Attributes:
        dataset_config (DatasetConfig): Configuration for dataset inputs and outputs.
    """

    dataset_config: DatasetConfig


class PromptTunerRunConfig(BaseRunConfig):
    """
    Configuration for running a prompt tuning task.

    Attributes:
        task_type (PromptTunerTaskType): The type of prompt tuning task to be executed.
        prompt_tuner_config (PromptTunerConfig): Configuration for the prompt tuning process.
    """

    task_type: PromptTunerTaskType
    prompt_tuner_config: PromptTunerConfig
