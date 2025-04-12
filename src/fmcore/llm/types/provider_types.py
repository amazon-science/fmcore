from abc import ABC
from typing import Union, Dict

from bears.util import Registry

from fmcore.llm.mixins.provider_mixins import AWSAccountMixin, APIKeyServiceMixin
from fmcore.types.mixins_types import RateLimiterMixin, RetryConfigMixin
from fmcore.types.typed import MutableTyped


class BaseProviderParams(MutableTyped, Registry, ABC):
    """
    Abstract base class for provider configurations.

    Extends Registry to allow dynamic registration of custom provider parameters
    required for llms.
    """

    @classmethod
    def from_dict(cls, provider_type: str, provider_params: Dict) -> "BaseProviderParams":
        """
        Creates an instance of the appropriate evaluator parameter subclass from a dictionary.

        This method is required by Pydantic validators to dynamically select parameter classes
        that are registered at runtime. This approach avoids the compile-time limitations of
        Pydantic discriminators.

        Args:
            provider_type (str): The type of provider to determine the correct subclass.
            provider_params (Dict): A dictionary of parameters for the llm.

        Returns:
            BaseEvaluatorParams: An instance of the resolved evaluator parameter subclass.
        """
        BaseProviderParamsClass = BaseProviderParams.get_subclass(key=provider_type)
        return BaseProviderParamsClass(**provider_params)


class BedrockProviderParams(BaseProviderParams, AWSAccountMixin, RateLimiterMixin, RetryConfigMixin):
    """
    Configuration for a Bedrock provider using AWS.

    This class combines AWS account settings with request configuration parameters
    (such as rate limits and retry policies) needed to interact with Bedrock services.
    It mixes in AWS-specific account details, rate limiting, and retry configurations
    to form a complete provider setup.

    Mixes in:
        AWSAccountMixin: Supplies AWS-specific account details (e.g., role ARN, region).
        RateLimiterMixin: Supplies API rate limiting settings.
        RetryConfigMixin: Supplies retry policy settings.
    """

    aliases = ["BEDROCK"]


class LambdaProviderParams(BaseProviderParams, AWSAccountMixin, RateLimiterMixin, RetryConfigMixin):
    """
    Configuration for a Bedrock provider using AWS.

    This class combines AWS account settings with request configuration parameters
    (such as rate limits and retry policies) needed to interact with Bedrock services.
    It mixes in AWS-specific account details, rate limiting, and retry configurations
    to form a complete provider setup.

    Mixes in:
        AWSAccountMixin: Supplies AWS-specific account details (e.g., role ARN, region).
        RateLimiterMixin: Supplies API rate limiting settings.
        RetryConfigMixin: Supplies retry policy settings.
    """

    aliases = ["LAMBDA"]

    function_arn: str
