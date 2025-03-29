from abc import ABC
from typing import Union

from fmcore.llm.enums.provider_enums import ProviderType
from fmcore.llm.mixins.provider_mixins import AWSAccountMixin, APIKeyServiceMixin
from fmcore.types.mixins_types import RateLimiterMixin, RetryConfigMixin
from fmcore.types.typed import MutableTyped


class BaseProviderParams(MutableTyped, ABC):
    """
    Abstract base class for provider configurations.

    This class defines the common interface and required attributes for all provider configuration
    classes. Its primary purpose is to support Pydantic's discriminator mechanism by ensuring that every
    concrete provider configuration includes a unique 'provider_type' field. This consistent contract
    allows Pydantic to automatically select the correct configuration model based on the value of
    'provider_type', enabling type-safe discrimination across different provider implementations.

    Attributes:
        provider_type (ProviderType): A unique identifier for the provider. Subclasses must override
            this field with a specific literal value corresponding to their provider (e.g., ProviderType.BEDROCK,
            ProviderType.OPENAI, etc.).
    """

    provider_type: ProviderType


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

    provider_type: ProviderType = ProviderType.BEDROCK


class LambdaProviderParams(BaseProviderParams, AWSAccountMixin, RateLimiterMixin, RetryConfigMixin):
    """
    Configuration for an AWS Lambda based Provider.

    This class encapsulates AWS account settings along with request configuration parameters
    (such as rate limits and retry policies) needed for invoking AWS Lambda functions.

    Attributes:
        function_name (str): The name of the Lambda function to be invoked.

    Mixes in:
        AWSAccountMixin: Provides AWS-specific account settings (e.g., role ARN, region).
        RateLimiterMixin: Provides API rate limiting settings.
        RetryConfigMixin: Provides retry policy settings.
    """

    provider_type: ProviderType = ProviderType.LAMBDA
    function_name: str


class OpenAIProviderParams(BaseProviderParams, APIKeyServiceMixin, RateLimiterMixin, RetryConfigMixin):
    """
    Configuration for an OpenAI provider using API-key authentication.

    This class combines API-key based service settings with request configuration parameters
    (such as rate limits and retry policies) needed to interact with OpenAI services via REST API.
    It mixes in API key settings, rate limiting, and retry configurations to form a complete provider setup.

    Mixes in:
        APIKeyServiceMixin: Provides the API key and an optional base URL for the service.
        RateLimiterMixin: Provides API rate limiting settings.
        RetryConfigMixin: Provides retry policy settings.
    """

    provider_type: ProviderType = ProviderType.OPENAI


# This union is used wherever provider parameters are required and helps in Pydantic discrimination.
# It should only contain provider parameter classes; adding anything else will break type consistency.
ProviderParamsUnion = Union[BedrockProviderParams, LambdaProviderParams, OpenAIProviderParams]
