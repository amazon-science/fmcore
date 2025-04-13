from datetime import timezone
from typing import Dict

import aioboto3
import boto3
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session

from fmcore.aws.constants import aws_constants as AWSConstants


class BotoFactory:
    """Factory to create and manage Boto3 clients with optional role-based authentication."""

    __clients: Dict[str, boto3.client] = {}

    @classmethod
    def __get_refreshable_session(cls, role_arn: str, region: str, session_name: str) -> get_session:
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
            """Refreshes credentials by assuming the specified role."""
            sts_client = boto3.client(AWSConstants.AWS_SERVICE_STS, region_name=region)
            response = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)
            credentials = response[AWSConstants.CREDENTIALS]
            return {
                AWSConstants.AWS_CREDENTIALS_ACCESS_KEY: credentials[AWSConstants.ACCESS_KEY_ID],
                AWSConstants.AWS_CREDENTIALS_SECRET_KEY: credentials[AWSConstants.SECRET_ACCESS_KEY],
                AWSConstants.AWS_CREDENTIALS_TOKEN: credentials[AWSConstants.SESSION_TOKEN],
                AWSConstants.AWS_CREDENTIALS_EXPIRY_TIME: credentials[AWSConstants.EXPIRATION].isoformat(),
            }

        # Create refreshable credentials
        refreshable_credentials = RefreshableCredentials.create_from_metadata(
            metadata=refresh(),
            refresh_using=refresh,
            method=AWSConstants.STS_ASSUME_ROLE_METHOD,
        )

        # Attach credentials to a botocore session
        botocore_session = get_session()
        botocore_session._credentials = refreshable_credentials
        botocore_session.set_config_variable(AWSConstants.REGION, region)

        return botocore_session

    @classmethod
    def __create_session(cls, *, role_arn: str = None, region: str, session_name: str) -> boto3.Session:
        """
        Creates a Boto3 session, either using role-based authentication or default credentials.

        Args:
            region (str): AWS region for the session.
            role_arn (str, optional): IAM role ARN to assume.
            session_name (str): Name for the session.

        Returns:
            boto3.Session: A configured Boto3 session.
        """
        if not role_arn:
            return boto3.Session(region_name=region)

        # Get a botocore session with refreshable credentials
        botocore_session = cls.__get_refreshable_session(
            role_arn=role_arn,
            region=region,
            session_name=session_name
        )

        return boto3.Session(botocore_session=botocore_session)

    @classmethod
    def get_client(cls, *, service_name: str, region: str, role_arn: str = None) -> boto3.client:
        """
        Retrieves a cached Boto3 client or creates a new one.

        Args:
            service_name (str): AWS service name (e.g., 's3', 'bedrock-runtime').
            region (str): AWS region for the client.
            role_arn (str, optional): IAM role ARN for authentication.

        Returns:
            boto3.client: A configured Boto3 client.
        """
        key = f"{service_name}-{region}-{role_arn or 'default'}"

        if key not in cls.__clients:
            session = cls.__create_session(
                region=region,
                role_arn=role_arn,
                session_name=f"{service_name}-Session"
            )
            cls.__clients[key] = session.client(service_name, region_name=region)

        return cls.__clients[key]

    @classmethod
    def get_async_session(cls, *,  service_name: str, region: str, role_arn: str = None) -> aioboto3.Session:
        session_name: str = f"Async-{service_name}-Session"

        def refresh():
            sts_client = boto3.client("sts", region_name=region)
            creds = sts_client.assume_role(RoleArn=role_arn, RoleSessionName=session_name)["Credentials"]
            return {
                "access_key": creds["AccessKeyId"],
                "secret_key": creds["SecretAccessKey"],
                "token": creds["SessionToken"],
                "expiry_time": creds["Expiration"].astimezone(timezone.utc).isoformat(),
            }

        creds = RefreshableCredentials.create_from_metadata(
            metadata=refresh(),
            refresh_using=refresh,
            method="sts-assume-role"
        )

        frozen = creds.get_frozen_credentials()

        session = aioboto3.Session(
            aws_access_key_id=frozen.access_key,
            aws_secret_access_key=frozen.secret_key,
            aws_session_token=frozen.token,
            region_name=region
        )

        return session