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


class RetryConfig(MutableTyped):
    """Defines retry parameters for API requests.

    Attributes:
        max_retries (int): Maximum number of retry attempts.
        backoff_factor (float): Factor by which the delay between retries increases (default: 1.0).
    """

    max_retries: int = Field(default=3)
    backoff_factor: float = Field(default=1.0)
    jitter: float = Field(default=1.0)
