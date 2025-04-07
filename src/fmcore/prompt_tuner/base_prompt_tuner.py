from datetime import datetime

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, NoReturn

from bears import FileMetadata, Writer

from fmcore.prompt_tuner.types.prompt_tuner_types import PromptTunerConfig, PromptTunerResult
from fmcore.types.enums.dataset_enums import DatasetType
from fmcore.types.typed import MutableTyped
from bears.util import Registry

from fmcore.utils.dataset_utils import DatasetUtils


class BasePromptTuner(MutableTyped, Registry, ABC):
    """
    Abstract base class for a prompt tuner. This class provides the
    necessary structure for creating specific prompt tuners based on
    the framework and configuration provided.

    Attributes:
        config (PromptTunerConfig): Configuration for the prompt tuner.
    """

    config: PromptTunerConfig

    @classmethod
    def of(cls, config: PromptTunerConfig) -> "BasePromptTuner":
        """
        Factory method to instantiate a specific prompt tuner based on the
        provided configuration.

        Args:
            config (PromptTunerConfig): Configuration used to instantiate the
            appropriate prompt tuner.

        Returns:
            BasePromptTuner: An instance of the correct subclass of BasePromptTuner
            based on the configuration's framework.
        """
        BasePromptTunerClass = BasePromptTuner.get_subclass(key=config.framework)
        return BasePromptTunerClass(config=config)

    @abstractmethod
    def tune_with_data(self, *, data: Dict[DatasetType, pd.DataFrame]) -> PromptTunerResult:
        """
        Abstract method to tune prompts using the provided dataset.

        Args:
            data (Dict[DatasetType, pd.DataFrame]): A dictionary mapping dataset types
            to pandas DataFrames containing the data used for prompt tuning.

        Returns:
            PromptTunerResult: The result of the prompt tuning process, containing
            the optimized prompts.
        """
        pass

    async def tune(self) -> NoReturn:
        """
        Loads datasets and performs the tuning process using the available configuration.

        This method loads the datasets as specified in the configuration, calls
        the abstract `tune_prompts_with_data` method to perform the tuning, and
        processes the resulting tuned prompts.
        """
        data: Dict[DatasetType, pd.DataFrame] = DatasetUtils.load_and_split_datasets(
            inputs=self.config.dataset_config.inputs
        )
        tuner_result: PromptTunerResult = self.tune_with_data(data=data)
        self.__process_results(tuner_result=tuner_result, output_metadata=self.config.dataset_config.output)

    def __process_results(self, tuner_result: PromptTunerResult, output_metadata: FileMetadata):
        """
        Process and save the tuned prompt results.

        Args:
            tuner_result (PromptTunerResult): Results containing optimized prompts.
            output_metadata (FileMetadata): Metadata specifying output location and format.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = f"{output_metadata.path.rstrip('/')}/{timestamp}/"

        prompt_records = [
            {
                "prompt_id": prompt.prompt_id,
                "prompt": prompt.prompt,
                "validation_score": prompt.validation_result.score if prompt.validation_result else None,
                "test_score": prompt.test_result.score if prompt.test_result else None,
            }
            for prompt in tuner_result.prompts
        ]
        prompts_df = pd.DataFrame(prompt_records)

        prompt_file_metadata = FileMetadata(
            name="prompts", path=output_directory, format=output_metadata.format
        )
        writer: Writer = Writer.of(file_format=prompt_file_metadata.format)
        writer.write(destination=prompt_file_metadata, data=prompts_df)
