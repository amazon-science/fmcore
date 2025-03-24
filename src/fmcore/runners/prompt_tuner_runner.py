from datetime import datetime
from typing import Dict, NoReturn

import pandas as pd
from bears import FileMetadata, Writer
from bears.constants import FileFormat
from bears.writer import ConfigWriter

from fmcore.prompt_tuner.base_prompt_tuner import BasePromptTuner
from fmcore.runners.base_runner import BaseRunner
from fmcore.types.enums.dataset_enums import DatasetType
from fmcore.types.prompt_tuner_types import PromptTunerResult
from fmcore.types.run_config_types import PromptTunerRunConfig
from fmcore.utils.dataset_utils import DatasetUtils


class PromptTunerRunner(BaseRunner):
    def run(self, run_config: dict) -> NoReturn:
        """
        Run the prompt tuner with the provided configuration.

        Args:
            run_config: Configuration for the prompt tuner run
        """
        config: PromptTunerRunConfig = PromptTunerRunConfig(**run_config)

        # Load and split datasets as needed
        data: Dict[DatasetType, pd.DataFrame] = DatasetUtils.load_and_split_datasets(
            inputs=config.dataset_config.inputs
        )

        # Run the prompt tuner
        prompt_tuner = BasePromptTuner.of(config=config.prompt_tuner_config)
        tuner_result: PromptTunerResult = prompt_tuner.tune(data=data)
        self.process_results(tuner_result=tuner_result, output_metadata=config.dataset_config.output)

    def process_results(self, tuner_result: PromptTunerResult, output_metadata: FileMetadata):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_directory = f"{output_metadata.path.rstrip('/')}/{timestamp}/"

        prompt_records = [
            {
                "prompt_id": prompt.prompt_id,
                "prompt": prompt.prompt,
                "validation_score": (
                    prompt.validation_result.score if prompt.validation_result else None
                ),
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