from typing import Union, Optional, List

from fmcore.types.typed import MutableTyped
from fmcore.llm.types.provider_types import (
    BedrockProviderParams,
    LambdaProviderParams,
    OpenAIProviderParams,
    ProviderParamsUnion,
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
    model_params: ModelParams = ModelParams()
    provider_params: ProviderParamsUnion


class DistributedLLMConfig(MutableTyped):
    """
    Configuration for distributed LLM execution across multiple providers.

    Why is this separate from LLMConfig?
    ------------------------------------
    1. **Type Consistency**: LLMConfig supports a single provider, while this class
       handles multiple. Merging them would require type checks everywhere.
    2. **Different Use Cases**: Single-provider configs are simple, while distributed
       execution involves multiple accounts, load balancing, and failover.

    Attributes:
        model_id (str): The identifier for the model.
        model_params (ModelParams): Parameters specific to the model.
        provider_params_list (List[ProviderParamsUnion]): A list of configurations for multiple providers.
    """

    model_id: str
    model_params: ModelParams = ModelParams()
    provider_params_list: List[ProviderParamsUnion]  # Holds configurations for multiple providers
