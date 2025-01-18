import types
from contextlib import contextmanager
from typing import *

from pydantic.typing import Literal


@contextmanager
def optional_dependency(
        *names: Union[List[str], str],
        error: Literal['raise', 'warn', 'ignore'] = "ignore",
        warn_every_time: bool = False,
        __WARNED_OPTIONAL_MODULES: Set[str] = set()  ## "Private" argument
) -> Optional[Union[Tuple[types.ModuleType, ...], types.ModuleType]]:
    """
    A contextmanager (used with "with") which passes code if optional dependencies are not present.
    Ref: https://stackoverflow.com/a/73838546/4900327

    Parameters
    ----------
    names: str or list of strings.
        The module name(s) which are optional.
    error: str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found in the "with" block:
        * raise : Raise an ImportError.
        * warn: print a warning (see `warn_every_time`).
        * ignore: do nothing.
    warn_every_time: bool
        Whether to warn every time an import is tried. Only applies when error="warn".
        Setting this to True will result in multiple warnings if you try to
        import the same library multiple times.

    Usage
    -----
    ## 1. Only run code if modules exist, otherwise ignore:
        with optional_dependency("pydantic", "sklearn", error="ignore"):
            from pydantic import BaseModel
            from sklearn.metrics import accuracy_score
            class AccuracyCalculator(BaseModel):
                decimals: int = 5
                def calculate(self, y_pred: List, y_true: List) -> float:
                    return round(accuracy_score(y_true, y_pred), self.decimals)
            print("Defined AccuracyCalculator in global context")
        print("Will be printed finally")  ## Always prints

    ## 2. Print warnings with error="warn". Multiple warings are be printed via `warn_every_time=True`.
        with optional_dependency("pydantic", "sklearn", error="warn"):
            from pydantic import BaseModel
            from sklearn.metrics import accuracy_score
            class AccuracyCalculator(BaseModel):
                decimals: int = 5
                def calculate(self, y_pred: List, y_true: List) -> float:
                    return round(accuracy_score(y_true, y_pred), self.decimals)
            print("Defined AccuracyCalculator in global context")
        print("Will be printed finally")  ## Always prints

    ## 3. Raise ImportError warnings with error="raise":
        with optional_dependency("pydantic", "sklearn", error="raise"):
            from pydantic import BaseModel
            from sklearn.metrics import accuracy_score
            class AccuracyCalculator(BaseModel):
                decimals: int = 5
                def calculate(self, y_pred: List, y_true: List) -> float:
                    return round(accuracy_score(y_true, y_pred), self.decimals)
            print("Defined AccuracyCalculator in global context")
        print("Will be printed finally")  ## Always prints
    """
    assert error in {"raise", "warn", "ignore"}
    names: Optional[Set[str]] = set(names)
    try:
        yield None
    except (ImportError, ModuleNotFoundError) as e:
        missing_module: str = e.name
        if len(names) > 0 and missing_module not in names:
            raise e  ## A non-optional dependency is missing
        if error == "raise":
            raise e
        if error == "warn":
            if missing_module not in __WARNED_OPTIONAL_MODULES or warn_every_time is True:
                msg = f'Missing optional dependency "{missing_module}". Use pip or conda to install.'
                print(f'Warning: {msg}')
                __WARNED_OPTIONAL_MODULES.add(missing_module)


_IS_RAY_INSTALLED: bool = False

with optional_dependency('ray'):
    import ray

    _IS_RAY_INSTALLED: bool = True


def _check_is_ray_installed():
    if not _IS_RAY_INSTALLED:
        raise ImportError(f'Dependency "ray" is not installed.')


_IS_DASK_INSTALLED: bool = False

DaskDataFrame = 'DaskDataFrame'
DaskSeries = 'DaskSeries'
with optional_dependency('dask'):
    import dask
    from dask.dataframe import DataFrame as DaskDataFrame, Scalar as DaskScalar, Series as DaskSeries

    assert isinstance(DaskDataFrame, type)
    assert isinstance(DaskSeries, type)

    _IS_DASK_INSTALLED: bool = True


def _check_is_dask_installed():
    if not _IS_DASK_INSTALLED:
        raise ImportError(f'Dependency "dask" is not installed.')


_IS_TORCH_INSTALLED: bool = False

with optional_dependency('torch'):
    import torch

    _IS_TORCH_INSTALLED: bool = True


def _check_is_torch_installed():
    if not _IS_TORCH_INSTALLED:
        raise ImportError(f'Dependency "torch" is not installed.')
