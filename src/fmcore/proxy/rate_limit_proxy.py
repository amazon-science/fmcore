from fmcore.proxy.base_proxy import BaseProxy
from aiolimiter import AsyncLimiter


class RateLimitedProxy(BaseProxy):
    """
    A proxy class that enhances clients with rate limiting capabilities.

    Attributes:
        rate_limiter (AsyncLimiter): An instance of AsyncLimiter to control the rate of requests.
    """

    # TODO: Generify this ratelimiter using custom rate limiter interfaces
    rate_limiter: AsyncLimiter
