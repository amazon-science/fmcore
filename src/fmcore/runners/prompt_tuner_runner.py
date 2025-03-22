from typing import Dict, NoReturn

import pandas as pd
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
        data: Dict[DatasetType, pd.DataFrame] = DatasetUtils.load_and_split_datasets(inputs=config.dataset_config.inputs)
        
        # Run the prompt tuner
        prompt_tuner = BasePromptTuner.of(config=config.prompt_tuner_config)
        tuner_result: PromptTunerResult = prompt_tuner.tune(data=data)

