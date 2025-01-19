from typing import *

import numpy as np
import pandas as pd

from ._alias import Alias
from ._math import is_int_in_floats_clothing
from ._pbar import ProgressBar
from ._structs import is_dict_like, is_set_like


def irange(low: Union[float, int], high: Union[float, int], step: Union[float, int] = 1):
    """Inclusive range, useful for coding up math notation."""
    if not (isinstance(low, int) or (isinstance(low, float) and low.is_integer())):
        raise ValueError(f"low={low} is not a valid integer.")
    if not (isinstance(high, int) or (isinstance(high, float) and high.is_integer())):
        raise ValueError(f"high={high} is not a valid integer.")
    if not (isinstance(step, int) or (isinstance(step, float) and step.is_integer())):
        raise ValueError(f"step={step} is not a valid integer.")
    return range(int(low), int(high) + 1, int(step))


def frange(low: float, high: float, step: float, *, limits: bool = True) -> List[float]:
    """Inclusive range, useful for coding up math notation."""
    assert isinstance(low, (int, float)) and isinstance(high, (int, float)) and isinstance(step, (int, float))
    out: List[float] = [
        x
        for x in [round(float(x) / step, 0) * step for x in np.arange(low, high + step, step)]
        if low <= x <= high
    ]
    if limits:
        out: List[float] = sorted(set(out).union({low, high}))
    return out


def is_valid_idx(
    l: Union[List, Tuple, np.ndarray, pd.Series, pd.DataFrame],
    idx: int,
    *,
    raise_error: bool = True,
) -> bool:
    assert isinstance(l, (list, tuple, np.ndarray, pd.Series, pd.DataFrame))
    assert idx >= 0, "Can only check validity of non-negative indexes"
    if len(l) == 0:
        if raise_error:
            raise ValueError(f"Cannot check validity of index for empty {str(type(l))}")
        return False  ## No index is valid
    return idx in range(0, len(l))


def iter_batches(
    struct: Union[List, Tuple, Set, Dict, np.ndarray, pd.Series, pd.DataFrame, int], batch_size: int, **kwargs
) -> Generator[List[Any], None, None]:
    assert isinstance(batch_size, int) and batch_size > 0
    progress_bar: Optional[Dict] = Alias.get_progress_bar(kwargs)
    if is_int_in_floats_clothing(struct):
        struct: List[int] = list(range(int(struct)))
    if is_set_like(struct):
        struct_type: Type = set
    elif is_dict_like(struct):
        struct_type: Type = dict
    else:
        struct_type: Optional[Type] = None

    struct_len: int = len(struct)
    pbar: ProgressBar = ProgressBar.of(
        progress_bar,
        total=struct_len,
        initial=0,
        desc="Iterating",
        prefer_kwargs=False,
        unit="item",
    )
    try:
        if struct_type is not None:
            buf: List[Any] = []
            if isinstance(struct, dict):
                struct: ItemsView = struct.items()
            for x in struct:
                buf.append(x)
                if len(buf) == batch_size:
                    out = struct_type(buf)
                    yield out
                    pbar.update(len(out))
                    buf: List[Any] = []
            if len(buf) > 0:
                out = struct_type(buf)
                yield out
                pbar.update(len(out))
        else:
            for i in range(0, struct_len, batch_size):
                if isinstance(struct, (pd.Series, pd.DataFrame)):
                    out = struct.iloc[i : min(i + batch_size, struct_len)]
                else:
                    out = struct[i : min(i + batch_size, struct_len)]
                yield out
                pbar.update(len(out))
        pbar.success()
    except Exception as e:
        pbar.failed()
        raise e
