from typing import Union, Optional, List, Dict

from pydantic import model_validator, SerializeAsAny

from fmcore.types.typed import MutableTyped
from fmcore.llm.types.provider_types import (
    BedrockProviderParams,
    BaseProviderParams,
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
        provider_type (str): The provider for the model.
        model_id (str): The identifier for the model.
        model_params (ModelParams): Parameters specific to the model.
        provider_params (BaseProviderParams): The parameters for the selected provider.

    Note:
        The `provider_type` is a string (instead of an enum) to allow user extensibility. Enums are closed by
        design — users cannot add new values without modifying the library code, which violates the Open-Closed
        Principle (OCP). By using strings, we support a registry-based pattern where users can register custom
        providers dynamically, enabling flexible and pluggable architectures.

        The `provider_params` field can point to a user-defined subclass of `BaseProviderParams`, typically
        resolved dynamically via a registry. To ensure that all fields from custom implementations are retained
        during serialization or model dumping, it should be wrapped with `SerializeAsAny`. Without it, fields
        not explicitly declared in `BaseProviderParams` would be silently dropped when the config is serialized.
    """

    provider_type: str
    model_id: str
    model_params: ModelParams = ModelParams()
    provider_params: SerializeAsAny[BaseProviderParams]

    @model_validator(mode="before")
    def parse_provider_params(cls, values: Dict):
        """
        Transforms provider_params based on provider_type before object creation.

        This method allows clients to register their providers dynamically at runtime.
        Each providers can have its own params.

        Args:
            values (Dict): The input dictionary containing provider_type and provider_params.

        Returns:
            Dict: The transformed values with provider_params converted to the appropriate class.
        """
        if cls is not LLMConfig:
            return values

        if isinstance(values.get("provider_params"), Dict):  # Only transform if it's a dict
            values["provider_params"] = BaseProviderParams.from_dict(
                provider_type=values.get("provider_type"), provider_params=values.get("provider_params")
            )
        return values


class DistributedLLMConfig(MutableTyped):
    """
    Configuration for distributed LLM execution across multiple providers.

    Why is this separate from LLMConfig?
    ------------------------------------
    1. **Functional Separation**: `LLMConfig` supports a single provider, while this
       class handles multiple providers for distributed use cases.

    2. **Clarity and Maintainability**: Avoids complex type branching and keeps
       each class focused, clean, and easier to work with.

    3. **Liskov Substitution Principle (LSP)**: Code expecting a single-provider config
       should not need to account for distributed behavior. This separation ensures
       substitutability without breaking expectations.

    Attributes:
        provider_type (str): The provider for the model.
        model_id (str): The identifier for the model.
        model_params (ModelParams): Parameters specific to the model.
        provider_params_list (List[BaseProviderParams]): A list of configurations for multiple provider params.

    Note:
        Follows the same principles as `LLMConfig`.

    """

    provider_type: str
    model_id: str
    model_params: ModelParams = ModelParams()
    provider_params_list: List[SerializeAsAny[BaseProviderParams]]

    @model_validator(mode="before")
    def parse_provider_params_list(cls, values: Dict):
        """
        Converts each provider_params item to its appropriate class using the given provider_type.
        """
        raw_params_list = values.get("provider_params_list", [])
        provider_type = values.get("provider_type")
        processed_params_list = []

        for raw_params in raw_params_list:
            if isinstance(raw_params, dict):
                processed_param = BaseProviderParams.from_dict(
                    provider_type=provider_type, provider_params=raw_params
                )
                processed_params_list.append(processed_param)
            else:
                processed_params_list.append(raw_params)  # Already an instance

        values["provider_params_list"] = processed_params_list
        return values
