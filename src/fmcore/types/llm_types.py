from typing import Union, Optional

from fmcore.types.typed import MutableTyped
from fmcore.types.provider_types import (
    BedrockProviderParams,
    LambdaProviderParams,
    OpenAIProviderParams,
)


class ModelParams(MutableTyped):
    """
    Represents common parameters used for configuring an LLM.

    Attributes:
        temperature (Optional[float]): Controls the randomness of the model's output.
        max_tokens (Optional[int]): Specifies the maximum number of tokens to generate in the response.
        top_p (Optional[float]): Enables nucleus sampling, where the model considers
            only the tokens comprising the top `p` cumulative probability mass.
    """

    temperature: Optional[float] = 0.5
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 0.5


class LLMConfig(MutableTyped):
    """
    Configuration for different LLM providers.

    Attributes:
        provider_params (Union[BedrockProviderParams, LambdaProviderParams, OpenAIProviderParams]):
            The parameters for the selected provider.
    """

    model_id: str
    provider_params: Union[BedrockProviderParams, LambdaProviderParams, OpenAIProviderParams]
    model_params: ModelParams = ModelParams()
