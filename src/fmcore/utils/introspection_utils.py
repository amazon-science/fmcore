import inspect
from typing import Any, Callable, Dict, Union, Type


class IntrospectionUtils:
    @staticmethod
    def filter_params(func: Union[Callable, Type], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filters the provided dictionary to include only the keys accepted by the given callable
        (either a function or a class constructor).

        If a class is provided, it inspects the __init__ method and ignores the first parameter
        (typically "self" or "cls").

        Args:
            func (Union[Callable, Type]): The target function or class.
            params (Dict[str, Any]): A dictionary of parameters to filter.

        Returns:
            Dict[str, Any]: A new dictionary containing only the parameters accepted by the callable.
        """
        if inspect.isclass(func):
            # Get the signature of the __init__ method and ignore the first parameter.
            sig = inspect.signature(func.__init__)
            allowed_keys = list(sig.parameters.keys())
            if allowed_keys and allowed_keys[0] in ("self", "cls"):
                allowed_keys = allowed_keys[1:]
        else:
            sig = inspect.signature(func)
            allowed_keys = list(sig.parameters.keys())

        return {key: value for key, value in params.items() if key in allowed_keys}
