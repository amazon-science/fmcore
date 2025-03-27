from typing import List
from fmcore.types.enums.provider_enums import ProviderType
from fmcore.types.mixins_types import AWSAccountMixin, RequestConfigMixin, APIKeyServiceMixin
from fmcore.types.typed import MutableTyped


class BedrockAccountConfig(AWSAccountMixin, RequestConfigMixin):
    """
    Configuration for a Bedrock account based on AWS.

    This class combines AWS account settings with request configuration settings (such as rate limits,
    timeouts, and retries) needed to interact with Bedrock services.

    Inherits:
        AWSAccountMixin: Provides AWS-specific configuration (e.g., role ARN, region).
        RequestConfigMixin: Provides API request-related settings.
    """

    pass


class LambdaAccountConfig(AWSAccountMixin, RequestConfigMixin):
    """
    Configuration for a Lambda account based on AWS.

    This class combines AWS account settings with request configuration settings necessary for invoking
    AWS Lambda functions.

    Attributes:
        function_name (str): The name of the Lambda function to be invoked.

    Inherits:
        AWSAccountMixin: Provides AWS-specific configuration.
        RequestConfigMixin: Provides API request-related settings.
    """

    function_name: str


class OpenAIAccountConfig(APIKeyServiceMixin, RequestConfigMixin):
    """
    Configuration for an OpenAI account based on API-key authentication.

    This class merges API-key based service settings with request configuration settings required
    to make REST API calls to OpenAI services.

    Inherits:
        APIKeyServiceMixin: Provides API key and optional base URL for the service.
        RequestConfigMixin: Provides API request-related settings.
    """

    pass


class BedrockProviderParams(MutableTyped):
    """
    Provider configuration parameters for Bedrock.

    This class specifies the provider type and the associated Bedrock account configurations.

    Attributes:
        provider_type (ProviderType): The type of the provider, fixed to ProviderType.BEDROCK.
        accounts (List[BedrockAccountConfig]): A list of Bedrock account configurations.
    """

    provider_type: ProviderType = ProviderType.BEDROCK
    accounts: List[BedrockAccountConfig]


class LambdaProviderParams(MutableTyped):
    """
    Provider configuration parameters for AWS Lambda.

    This class specifies the provider type and the associated Lambda account configurations.

    Attributes:
        provider_type (ProviderType): The type of the provider, fixed to ProviderType.LAMBDA.
        accounts (List[LambdaAccountConfig]): A list of Lambda account configurations.
    """

    provider_type: ProviderType = ProviderType.LAMBDA
    accounts: List[LambdaAccountConfig]


class OpenAIProviderParams(MutableTyped):
    """
    Provider configuration parameters for OpenAI.

    This class specifies the provider type and the associated OpenAI account configurations.

    Attributes:
        provider_type (ProviderType): The type of the provider, fixed to ProviderType.OPENAI.
        accounts (List[OpenAIAccountConfig]): A list of OpenAI account configurations.
    """

    provider_type: ProviderType = ProviderType.OPENAI
    accounts: List[OpenAIAccountConfig]
