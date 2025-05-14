import boto3

from fmcore.aws.constants import aws_constants as AWSConstants


def assume_role_and_get_credentials(role_arn: str, region: str, session_name: str) -> dict:
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
