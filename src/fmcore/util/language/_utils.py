from typing import *

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar as pd_is_scalar


def get_default(*vals) -> Optional[Any]:
    for x in vals:
        if not is_null(x):
            return x
    return None


def unset(obj, attr_name: str, new_val: Any = None, delete: bool = True):
    attr: Any = getattr(obj, attr_name)
    setattr(obj, attr_name, new_val)
    if delete:
        del attr


def get_true(*vals) -> bool:
    for x in vals:
        if x is True:
            return x
    return False


if_else = lambda cond, x, y: (x if cond is True else y)  ## Ternary operator
is_series = lambda x: isinstance(x, pd.Series)
is_df = lambda x: isinstance(x, pd.DataFrame)


## ======================== None utils ======================== ##
def any_are_none(*args) -> bool:
    for x in args:
        if x is None:
            return True
    return False


def all_are_not_none(*args) -> bool:
    return not any_are_none(*args)


def all_are_none(*args) -> bool:
    for x in args:
        if x is not None:
            return False
    return True


def any_are_not_none(*args) -> bool:
    return not all_are_none(*args)


def all_are_true(*args) -> bool:
    for x in args:
        assert x in {True, False}
        if not x:  ## Check for falsy values
            return False
    return True


def all_are_false(*args) -> bool:
    for x in args:
        assert x in {True, False}
        if x:  ## Check for truthy values
            return False
    return True


def none_count(*args) -> int:
    none_count: int = 0
    for x in args:
        if x is None:
            none_count += 1
    return none_count


def not_none_count(*args) -> int:
    return len(args) - none_count(*args)


def multiple_are_none(*args) -> bool:
    return none_count(*args) >= 2


def multiple_are_not_none(*args) -> bool:
    return not_none_count(*args) >= 2


def equal(*args) -> bool:
    if len(args) == 0:
        raise ValueError("Cannot find equality for zero arguments")
    if len(args) == 1:
        return True
    first_arg = args[0]
    for arg in args[1:]:
        if arg != first_arg:
            return False
    return True


def is_scalar(x: Any, method: Literal["numpy", "pandas"] = "pandas") -> bool:
    if method == "pandas":
        ## Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.is_scalar.html
        ## Actual code: github.com/pandas-dev/pandas/blob/0402367c8342564538999a559e057e6af074e5e4/pandas/_libs/lib.pyx#L162
        return pd_is_scalar(x)
    if method == "numpy":
        ## Ref: https://numpy.org/doc/stable/reference/arrays.scalars.html#built-in-scalar-types
        return np.isscalar(x)
    raise NotImplementedError(f'Unsupported method: "{method}"')


is_null = lambda z: pd.isnull(z) if is_scalar(z) else (z is None)
is_not_null = lambda z: not is_null(z)


class Utility:
    def __init__(self):
        raise TypeError(f'Cannot instantiate utility class "{str(self.__class__)}"')
