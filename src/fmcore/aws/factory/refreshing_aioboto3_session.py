import aioboto3
from datetime import datetime, timedelta, timezone

from fmcore.aws.constants import aws_constants as AWSConstants
from fmcore.aws.factory.boto_utils import assume_role_and_get_credentials

REFRESH_MARGIN = timedelta(minutes=5)

class RefreshingAioboto3Session:
    def __init__(self, aioboto3_session: aioboto3.Session):
        self._creds = None
        self._expiry = None
        self._session = aioboto3_session

    async def _refresh_credentials(self, session_name: str,region_name: str,  role_arn: str = None):
        print("Refreshing credentials...")
        creds = assume_role_and_get_credentials(role_arn, region_name, session_name)
        self._creds = {
            AWSConstants.AWS_ACCESS_KEY_ID: creds[AWSConstants.AWS_CREDENTIALS_ACCESS_KEY],
            AWSConstants.AWS_SECRET_ACCESS_KEY: creds[AWSConstants.AWS_CREDENTIALS_SECRET_KEY],
            AWSConstants.AWS_SESSION_TOKEN: creds[AWSConstants.AWS_CREDENTIALS_TOKEN]
        }
        self._expiry = creds[AWSConstants.AWS_CREDENTIALS_EXPIRY_TIME].replace(tzinfo=timezone.utc)



    async def get_client(self, service_name: str, region_name: str, role_arn: str = None):
        session_name: str = self._session.name
        now = datetime.now(timezone.utc)
        if not self._creds or now + REFRESH_MARGIN >= self._expiry:
            await self._refresh_credentials(session_name, region_name, role_arn)

        return self._session.client(
            service_name,
            region_name=region_name,
            **self._creds,
        )
