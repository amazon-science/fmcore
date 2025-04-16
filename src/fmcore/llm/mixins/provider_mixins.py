from typing import Optional

from pydantic import Field

from fmcore.aws.enums.aws_enums import AWSRegion
from fmcore.types.mixins_types import Mixin
from fmcore.types.typed import MutableTyped


class AWSAccountMixin(MutableTyped, Mixin):
    """
    Mixin for AWS account configuration, including IAM role and region.

    Attributes:
        role_arn (str): The IAM role ARN to assume for accessing AWS services.
        region (str): The AWS region where the account operates. Defaults to 'us-east-1'.
    """

    role_arn: Optional[str] = Field(default=None)
    region: Optional[str] = Field(default="us-east-1")


class APIKeyServiceMixin(MutableTyped, Mixin):
    """
    Mixin for API-key based service configuration.

    Attributes:
        api_key (str): The API key used for authentication.
        base_url (Optional[str]): The base URL for API requests. Defaults to None.
    """

    api_key: str
    base_url: Optional[str] = None
