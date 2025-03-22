from abc import ABC
from typing import List, Optional

from fmcore.types.typed import MutableTyped
from fmcore.types.enums.provider_enums import ProviderType


class NetworkConfig(MutableTyped, ABC):
    """Abstract base class for network-related configurations.

    Attributes:
        rate_limit (int): The maximum number of requests allowed per time unit.
        timeout (int): The timeout duration for network requests (default: 30).
    """

    rate_limit: Optional[int] = 50
    timeout: Optional[int] = 30


class BedrockAccountConfig(NetworkConfig):
    """Configuration for a Bedrock account.

    Attributes:
        account_id (str): The unique identifier for the account (default: "default").
        region (str): The AWS region where the account is located (default: "us-east-1").
        role_arn (str): The IAM role ARN associated with the account (default: None).
    """

    region: Optional[str] = "us-east-1"
    role_arn: Optional[str] = None


class BedrockProviderParams(MutableTyped):
    """Parameters specific to the Bedrock provider.

    Attributes:
        provider_type (Literal[ProviderType.BEDROCK]): The provider type.
        accounts (List[BedrockAccountConfig]): A list of account configurations.
    """

    provider_type: ProviderType = ProviderType.BEDROCK
    accounts: List[BedrockAccountConfig]


class OpenAIAccountConfig(NetworkConfig):
    """Configuration for an OpenAI account.

    Attributes:
        api_key (str): The API key used for authentication with OpenAI.
        base_url (Optional[str]): The base URL for the OpenAI API (optional).
    """

    api_key: str
    base_url: Optional[str] = None


class OpenAIProviderParams(MutableTyped):
    """Parameters specific to the OpenAI provider.

    Attributes:
        provider_type (Literal[ProviderType.OPENAI]): The provider type.
        accounts (List[OpenAIAccountConfig]): A list of OpenAI account configurations.
    """

    provider_type: ProviderType = ProviderType.OPENAI
    accounts: List[OpenAIAccountConfig]


class LambdaAccountConfig(NetworkConfig):
    """Configuration for a Lambda account.

    Attributes:
        account_id (str): The unique identifier for the account (default: "default-account").
        region (str): The AWS region where the Lambda function is deployed (default: "us-east-1").
        role_arn (str): The IAM role ARN associated with the account (default: None).
        function_name (str): The name of the Lambda function (default: "default-function").
    """

    function_name: str
    region: str = "us-east-1"
    role_arn: Optional[str] = None


class LambdaProviderParams(MutableTyped):
    """Parameters specific to the Lambda provider.

    Attributes:
        provider_type (Literal[ProviderType.LAMBDA]): The provider type.
        accounts (List[LambdaAccountConfig]): A list of Lambda execution configurations.
    """

    provider_type: ProviderType = ProviderType.LAMBDA
    accounts: List[LambdaAccountConfig]
