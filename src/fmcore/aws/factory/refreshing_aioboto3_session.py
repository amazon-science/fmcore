from fmcore.types.typed import MutableTyped
import aioboto3
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
from dateutil.parser import parse

from fmcore.aws.constants import aws_constants as AWSConstants
from fmcore.aws.factory.boto_utils import assume_role_and_get_credentials

REFRESH_MARGIN = timedelta(minutes=5)


class RefreshingAioboto3Session(MutableTyped):
    session: aioboto3.Session
    creds: Optional[Dict[str, str]] = None
    expiry: Optional[datetime] = None

    async def _refresh_credentials(self, session_name: str, region_name: str, role_arn: str = None):
        creds = assume_role_and_get_credentials(role_arn, region_name, session_name)
        self.creds = {
            AWSConstants.AWS_ACCESS_KEY_ID: creds[AWSConstants.AWS_CREDENTIALS_ACCESS_KEY],
            AWSConstants.AWS_SECRET_ACCESS_KEY: creds[AWSConstants.AWS_CREDENTIALS_SECRET_KEY],
            AWSConstants.AWS_SESSION_TOKEN: creds[AWSConstants.AWS_CREDENTIALS_TOKEN],
        }
        expiry_str = creds[AWSConstants.AWS_CREDENTIALS_EXPIRY_TIME]
        expiry_dt = parse(expiry_str).astimezone(timezone.utc)
        self.expiry = expiry_dt

    @staticmethod
    def get_session_name(service_name: str):
        return f"Async-{service_name}-Session"

    async def get_client(self, service_name: str, region_name: str, role_arn: str = None):
        session_name: str = RefreshingAioboto3Session.get_session_name(service_name)
        now = datetime.now(timezone.utc)
        if not self.creds or now + REFRESH_MARGIN >= self.expiry:
            await self._refresh_credentials(session_name, region_name, role_arn)

        return self.session.client(
            service_name,
            region_name=region_name,
            **self.creds,
        )
