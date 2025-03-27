from typing import Optional
from pydantic import Field

from fmcore.types.config_types import RateLimitConfig
from fmcore.types.enums.aws_enums import AWSRegion
from fmcore.types.typed import MutableTyped


class Mixin:
    """Marker interface for mixin classes."""

    pass


class AWSAccountMixin(MutableTyped, Mixin):
    """
    Mixin for AWS account configuration, including IAM role and region.

    Attributes:
        role_arn (str): The IAM role ARN to assume for accessing AWS services.
        region (str): The AWS region where the account operates. Defaults to 'us-east-1'.
    """

    role_arn: str
    region: str = Field(default=AWSRegion.US_EAST_1.value)


class APIKeyServiceMixin(MutableTyped, Mixin):
    """
    Mixin for API-key based service configuration.

    Attributes:
        api_key (str): The API key used for authentication.
        base_url (Optional[str]): The base URL for API requests. Defaults to None.
    """

    api_key: str
    base_url: Optional[str] = None


class RequestConfigMixin(MutableTyped, Mixin):
    """
    Mixin for request-level configurations, including rate limits, timeouts, and retries.

    This mixin is designed for REST API configurations used by network providers
    (e.g., Bedrock, OpenAI). It centralizes settings that govern API request behaviors,
    such as rate limiting, timeout duration, and retry attempts.

    Attributes:
        rate_limit (Optional[RateLimitConfig]): The rate limit configuration to
            apply to API requests.
        timeout (int): The maximum allowed request time in seconds. Defaults to 300.
        retries (int): The number of retry attempts for failed API requests. Defaults to 3.
    """

    rate_limit: Optional[RateLimitConfig] = Field(default=RateLimitConfig)
    timeout: int = Field(default=300)
    retries: int = Field(default=3)
