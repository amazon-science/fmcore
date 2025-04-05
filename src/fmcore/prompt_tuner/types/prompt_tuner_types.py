from typing import List, Dict, Optional

import pandas as pd
from pydantic import model_validator

from fmcore.prompt_tuner.types.enums.prompt_tuner_enums import PromptTunerFramework, PromptTunerTaskType
from fmcore.prompt_tuner.types.optimizer_types import BaseOptimizerConfig
from fmcore.types.config_types import DatasetConfig
from fmcore.types.typed import MutableTyped


class PromptField(MutableTyped):
    """
    Represents a field in a prompt, including its name, description, and type.

    Attributes:
        name (str): The name of the field.
        description (str): A description of the field.
        field_type (str): The type of the field, default is "string".
    """

    name: str
    description: str
    field_type: str = "string"


class PromptConfig(MutableTyped):
    """
    Represents the configuration for a prompt, including the prompt string and its input/output fields.

    Attributes:
        prompt (str): The prompt string.
        input_fields (List[PromptField]): A list of input fields associated with the prompt.
        output_fields (List[PromptField]): A list of output fields associated with the prompt.
    """

    prompt: str
    input_fields: Optional[List[PromptField]] = []
    output_fields: Optional[List[PromptField]] = []


class PromptTunerConfig(MutableTyped):
    """
    Configuration class for a prompt tuner, including the framework, prompt configuration, and optimizer configuration.
    """

    task_type: PromptTunerTaskType
    dataset_config: DatasetConfig
    prompt_config: PromptConfig
    framework: PromptTunerFramework
    optimizer_config: BaseOptimizerConfig

    @model_validator(mode="before")
    def validate_prompt_tuner_config(cls, values: Dict):
        """
        Validates the configuration values before the model is created.

        1. Ensures that `input_fields` and `output_fields` are provided in `prompt_config` if the framework is 'dspy'.
        2. Converts `optimizer_config` to a `BaseOptimizerConfig` if it's a dictionary.

        Args:
            values (Dict): The input values for the class.

        Returns:
            Dict: The transformed and validated values.

        Raises:
            ValueError: If `input_fields` or `output_fields` are missing when the framework is 'dspy'.
        """
        framework = values.get("framework")
        prompt_config_data = values.get("prompt_config")

        # Convert prompt_config_data to an instance of PromptConfig if it's a dictionary
        if isinstance(prompt_config_data, dict):
            prompt_config = PromptConfig(**prompt_config_data)
            values["prompt_config"] = prompt_config
        else:
            prompt_config = prompt_config_data

        # Validate prompt fields if the framework is 'dspy'
        if framework == PromptTunerFramework.DSPY.value:
            if not prompt_config.input_fields or not prompt_config.output_fields:
                raise ValueError(
                    "For 'dspy' framework, both input_fields and output_fields must be provided in the prompt config."
                )
        elif framework == PromptTunerFramework.LMOPS.value:
            if prompt_config.input_fields or prompt_config.output_fields:
                raise ValueError(
                    "For 'lmops' framework, input_fields and output_fields should not be provided in the prompt config."
                )

        # Handle optimizer config transformation
        if isinstance(values.get("optimizer_config"), Dict):
            values["optimizer_config"] = BaseOptimizerConfig.from_dict(
                optimizer_config=values.get("optimizer_config")
            )

        return values


class PromptEvaluationResult(MutableTyped):
    """
    Represents the result of evaluating a prompt, including a score and optional data.

    Attributes:
        score (float): The evaluation score for the prompt.
        data (Optional[pd.DataFrame]): Optional additional data associated with the evaluation.
    """

    score: float  # TODO this should be metric name
    data: Optional[pd.DataFrame]


class TunedPrompt(MutableTyped):
    """
    Represents a tuned prompt, including its ID, prompt text, and evaluation results.

    Attributes:
        prompt_id (str): The unique identifier for the tuned prompt.
        prompt (str): The tuned prompt text.
        validation_result (Optional[PromptEvaluationResult]): The result of the validation evaluation.
        test_result (Optional[PromptEvaluationResult]): The result of the test evaluation.
    """

    prompt_id: str
    prompt: str
    validation_result: Optional[PromptEvaluationResult]
    test_result: Optional[PromptEvaluationResult]


class PromptTunerResult(MutableTyped):
    """
    Represents the result of a prompt tuning process, including a list of tuned prompts.

    Attributes:
        prompts (List[TunedPrompt]): A list of the tuned prompts resulting from the tuning process.
    """

    prompts: List[TunedPrompt]
