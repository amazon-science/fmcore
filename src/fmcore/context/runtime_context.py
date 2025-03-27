from typing import TypeVar, Generic
from aiolimiter import AsyncLimiter
from fmcore.types.typed import MutableTyped

T = TypeVar("T")


class RuntimeContext(MutableTyped, Generic[T]):
    """Represents the runtime context for managing a client with additional controls.

    This class serves as a container for a client instance along with related  runtime configurations such as
    rate limiting. It allows for structured management of client behavior at runtime.

    Attributes:
        client (T): The wrapped client instance of generic type T.
        rate_limiter (AsyncLimiter): An async rate limiter instance that controls the frequency of operations.
            This may later be replaced with a more generic rate limiter abstraction.

    Type Parameters:
        T: The type of the client being managed.
    """

    client: T  # Generic client instance
    rate_limiter: AsyncLimiter  # TODO: Generify this rate limiter using custom interfaces
