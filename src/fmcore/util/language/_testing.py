from itertools import product
from typing import *

from ._import import optional_dependency
from ._structs import flatten2d

with optional_dependency("parameterized"):
    from parameterized import parameterized

    def parameterized_name_func(test, _, param):
        ## Ref: https://kracekumar.com/post/618264170735009792/parameterize-python-tests/
        return f"{test.__name__}_{parameterized.to_safe_name('_'.join([str(x) for x in param.args]))}"


def parameterized_flatten(*args) -> List:
    return flatten2d(list(product(*args)))
