import backoff

from functools import wraps
from fmcore.types.config_types import RetryConfig
from fmcore.utils.logging_utils import Log


class RetryUtil:
    """
    Utility class for applying backoff-based retry logic using a custom RetryConfig.

    Methods:
        with_backoff(retry_config_getter): Decorator factory that applies retry logic
                                           using the RetryConfig from the instance.
    """

    @staticmethod
    def with_backoff(retry_config_getter):
        """
        Decorator factory that applies retry logic using the RetryConfig from the class instance.

        Args:
            retry_config_getter (Callable): A function that takes `self` and returns a RetryConfig.

        Returns:
            Callable: Decorator for the method to be retried.
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(self, *args, **kwargs):
                retry_config: RetryConfig = retry_config_getter(self)

                decorated = backoff.on_exception(
                    exception=Exception,
                    wait_gen=backoff.expo,
                    giveup=lambda e: not any(
                        exception in str(e) for exception in retry_config.retryable_exceptions
                    ),
                    max_tries=retry_config.max_retries,
                    logger=Log.get_logger(),
                )(func)

                return await decorated(self, *args, **kwargs)

            return wrapper

        return decorator
