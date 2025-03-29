from aiolimiter import AsyncLimiter
from fmcore.types.config_types import RateLimitConfig


class RateLimiterUtils:
    """
    Utility class for creating asynchronous rate limiters.

    This class groups functionality related to rate limiting, so you don't have to repeatedly
    import or write the logic for instantiating rate limiters. It uses the provided RateLimitConfig
    to create an AsyncLimiter instance.
    """

    @staticmethod
    def create_async_rate_limiter(rate_limit_config: RateLimitConfig) -> AsyncLimiter:
        """
        Creates an asynchronous rate limiter based on the provided rate limit configuration.

        Args:
            rate_limit_config (RateLimitConfig): Configuration parameters for rate limiting,
                including:
                  - max_rate: Maximum number of requests allowed.
                  - time_period: Time window (in seconds) within which the requests are counted.

        Returns:
            AsyncLimiter: An asynchronous rate limiter configured with the specified settings.
        """
        return AsyncLimiter(
            max_rate=rate_limit_config.max_rate,
            time_period=rate_limit_config.time_period,
        )
