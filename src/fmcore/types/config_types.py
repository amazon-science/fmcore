from pydantic import Field

from fmcore.types.typed import MutableTyped


class RateLimitConfig(MutableTyped):
    """Defines rate limiting parameters for API requests.

    Attributes:
        max_rate (int): Maximum number of requests allowed.
        time_period (int): Time window (in seconds) within which the requests are counted (default: 60s).
    """

    max_rate: int = Field(default=60)
    time_period: int = Field(default=60)
