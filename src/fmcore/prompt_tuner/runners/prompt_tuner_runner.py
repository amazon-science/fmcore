from datetime import datetime
from typing import Dict, NoReturn

import pandas as pd
from bears import FileMetadata, Writer

from fmcore.prompt_tuner.base_prompt_tuner import BasePromptTuner
from fmcore.prompt_tuner.types.prompt_tuner_run_config_types import PromptTunerRunConfig
from fmcore.prompt_tuner.types.prompt_tuner_types import PromptTunerResult
from fmcore.runners.base_runner import BaseRunner
from fmcore.types.enums.dataset_enums import DatasetType
from fmcore.utils.dataset_utils import DatasetUtils


class PromptTunerRunner(BaseRunner[PromptTunerRunConfig]):
    """
    Runner class for executing prompt tuning.

    This class loads datasets, runs the prompt tuner, and processes results.
    """

    async def run(self, config: PromptTunerRunConfig) -> NoReturn:
        """
        Execute the prompt tuner using the provided configuration.

        Args:
            config (PromptTunerRunConfig): Configuration containing dataset and tuning parameters.
        """
        # Load and split datasets
        data: Dict[DatasetType, pd.DataFrame] = DatasetUtils.load_and_split_datasets(
            inputs=config.dataset_config.inputs
        )

        # Run the prompt tuner
        prompt_tuner = BasePromptTuner.of(config=config.prompt_tuner_config)
        tuner_result: PromptTunerResult = prompt_tuner.tune(data=data)

        # Process and save results
        self.process_results(tuner_result=tuner_result, output_metadata=config.dataset_config.output)

    def process_results(self, tuner_result: PromptTunerResult, output_metadata: FileMetadata):
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
