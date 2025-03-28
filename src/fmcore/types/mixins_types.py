from typing import Optional
from pydantic import Field

from fmcore.types.config_types import RateLimitConfig, RetryConfig
from fmcore.types.enums.aws_enums import AWSRegion
from fmcore.types.typed import MutableTyped


class Mixin:
    """
    Marker class to indicate that this is a mixin.

    This class has no functional significance but helps static type analyzers,
    such as pylint and mypy, detect potential conflicts in method resolution
    when multiple mixins are used.

    Static analysis tools can use this marker to differentiate mixins from
    concrete classes, aiding in detecting conflicting methods or variables.
    """

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


class RateLimiterMixin(MutableTyped, Mixin):
    """
    Mixin for rate limiting configurations.

    Attributes:
        rate_limit (Optional[RateLimitConfig]): The rate limit configuration to
            apply to API requests.
    """

    rate_limit: Optional[RateLimitConfig] = Field(default=RateLimitConfig)


class RetryConfigMixin(MutableTyped, Mixin):
    """
    Mixin for retry configurations.

    Attributes:
        retries (int): The number of retry attempts for failed API requests. Defaults to 3.
    """

    retries: Optional[RetryConfig] = Field(default=RetryConfig)
