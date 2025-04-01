from typing import List, Dict, Optional

import pandas as pd
from pydantic import model_validator

from fmcore.prompt_tuner.types.enums.prompt_tuner_enums import PromptTunerFramework
from fmcore.prompt_tuner.types.optimizer_types import BaseOptimizerConfig
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
    input_fields: List[PromptField]
    output_fields: List[PromptField]


class PromptTunerConfig(MutableTyped):
    """
    Configuration class for a prompt tuner, including the framework, prompt configuration, and optimizer configuration.

    Attributes:
        framework (PromptTunerFramework): The framework used for tuning.
        prompt_config (PromptConfig): Configuration details for the prompt.
        optimizer_config (BaseOptimizerConfig): Configuration details for the optimizer.

    Methods:
        parse_optimizer_config (model_validator): A Pydantic validator that parses the optimizer config if it's provided as a dictionary.
    """

    framework: PromptTunerFramework
    prompt_config: PromptConfig
    optimizer_config: BaseOptimizerConfig

    @model_validator(mode="before")
    def parse_optimizer_config(cls, values: Dict):
        """
        Validates and transforms the optimizer configuration before the model is created.
        If the optimizer configuration is a dictionary, it converts it into a BaseOptimizerConfig object.

        Args:
            values (Dict): The input values for the class.

        Returns:
            Dict: The transformed values with the optimizer_config as a BaseOptimizerConfig.
        """
        if isinstance(values.get("optimizer_config"), Dict):  # Only transform if it's a dict
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
