from fmcore.experimental.proxy.base_proxy import BaseProxy
from aiolimiter import AsyncLimiter


class RateLimitedProxy(BaseProxy):
    """A proxy that adds rate limiting to a client."""

    rate_limiter: AsyncLimiter
