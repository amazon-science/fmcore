from typing import Dict

import aioboto3
import boto3
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session

from aiobotocore.session import AioSession
from aiobotocore.credentials import AioRefreshableCredentials

from fmcore.utils.string_utils import is_empty_or_none
from fmcore.aws.constants import aws_constants as AWSConstants
from fmcore.aws.factory.boto_utils import assume_role_and_get_credentials


class BotoFactory:
    """Factory to create and manage Boto3 clients with optional role-based authentication."""

    __clients: Dict[str, boto3.client] = {}

    @classmethod
    def __get_refreshable_session(cls, role_arn: str, region_name: str, session_name: str) -> boto3.Session:
        """
        Creates a botocore session with refreshable credentials for the assumed IAM role.

        Args:
            role_arn (str): ARN of the IAM role to assume.
            session_name (str): Name for the assumed session.
            region (str, optional): AWS region for the session..

        Returns:
            boto3.Session: A session with automatically refreshed credentials.
        """

        def refresh() -> dict:
            return assume_role_and_get_credentials(role_arn, region_name, session_name)

        # Create refreshable credentials
        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=refresh(),
            refresh_using=refresh,
            method=AWSConstants.STS_ASSUME_ROLE_METHOD,
        )

        # Attach credentials to a botocore session
        botocore_session = get_session()
        botocore_session._credentials = refreshable_credentials
        botocore_session.set_config_variable(AWSConstants.REGION, region_name)

        return botocore_session

    @classmethod
    def __create_session(cls, *, role_arn: str = None, region_name: str, session_name: str) -> boto3.Session:
        """
        Creates a Boto3 session, either using role-based authentication or default credentials.

        Args:
            region (str): AWS region for the session.
            role_arn (str, optional): IAM role ARN to assume.
            session_name (str): Name for the session.

        Returns:
            boto3.Session: A configured Boto3 session.
        """
        if is_empty_or_none(role_arn=role_arn):
            return boto3.Session(region_name=region_name)

        # Get a botocore session with refreshable credentials
        botocore_session = cls.__get_refreshable_session(
            role_arn=role_arn, region_name=region_name, session_name=session_name
        )

        return boto3.Session(botocore_session=botocore_session)

    @classmethod
    def get_client(cls, *, service_name: str, region_name: str, role_arn: str = None) -> boto3.client:
        """
        Retrieves a cached Boto3 client or creates a new one.

        Args:
            service_name (str): AWS service name (e.g., 's3', 'bedrock-runtime').
            region (str): AWS region for the client.
            role_arn (str, optional): IAM role ARN for authentication.

        Returns:
            boto3.client: A configured Boto3 client.
        """
        key = f"{service_name}-{region_name}-{role_arn or 'default'}"

        if key not in cls.__clients:
            session = cls.__create_session(
                region_name=region_name, role_arn=role_arn, session_name=f"{service_name}-Session"
            )
            cls.__clients[key] = session.client(service_name, region_name=region_name)

        return cls.__clients[key]

    @classmethod
    def __get_refreshable_async_session(
        cls, role_arn: str, region_name: str, session_name: str
    ) -> AioSession:
        """
        Creates a botocore session with refreshable credentials for the assumed IAM role.

        Args:
            role_arn (str): ARN of the IAM role to assume.
            session_name (str): Name for the assumed session.
            region (str, optional): AWS region for the session..

        Returns:
            boto3.Session: A session with automatically refreshed credentials.
        """

        def refresh() -> dict:
            return assume_role_and_get_credentials(role_arn, region_name, session_name)

        # Create refreshable credentials
        refreshable_credentials = AioRefreshableCredentials.create_from_metadata(
            metadata=refresh(),
            refresh_using=refresh,
            method=AWSConstants.STS_ASSUME_ROLE_METHOD,
        )

        # Attach credentials to a botocore session
        botocore_session = AioSession()
        botocore_session._credentials = refreshable_credentials
        botocore_session.set_config_variable(AWSConstants.REGION, region_name)

        return botocore_session

    @classmethod
    def get_async_session(
        cls, *, service_name: str, region_name: str, role_arn: str = None
    ) -> aioboto3.Session:
        if is_empty_or_none(role_arn=role_arn):
            return aioboto3.Session(region_name=region_name)

        botocore_session = cls.__get_refreshable_async_session(
            role_arn=role_arn, region_name=region_name, session_name=f"Async-{service_name}-Session"
        )

        return aioboto3.Session(botocore_session=botocore_session)
