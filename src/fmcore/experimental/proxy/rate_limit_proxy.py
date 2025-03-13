from fmcore.experimental.proxy.base_proxy import BaseProxy
from aiolimiter import AsyncLimiter
from typing import Optional


class RateLimitedProxy(BaseProxy):
    """
    A proxy class that enhances clients with rate limiting capabilities.

    Attributes:
        rate_limiter (AsyncLimiter): An instance of AsyncLimiter to control the rate of requests.
    """

    rate_limiter: AsyncLimiter
