from typing import *

import numpy as np
import pandas as pd
from pydantic.typing import Literal

from ._structs import is_numpy_float_array

is_even = lambda x: x % 2 == 0
is_odd = lambda x: x % 2 == 1

is_int_in_floats_clothing = lambda x: isinstance(x, int) or (isinstance(x, float) and int(x) == x)


def mean(vals):
    return sum(vals) / len(vals)


def clip(low: Union[int, float], val: Union[int, float], high: Union[int, float]):
    assert isinstance(low, (int, float, np.integer, np.float_))
    assert isinstance(high, (int, float, np.integer, np.float_))
    assert isinstance(val, (int, float, np.integer, np.float_))
    assert low <= high
    return max(low, min(val, high))


def pad_interval(low: Union[int, float], high: Union[int, float], pad: float) -> Tuple[float, float]:
    assert isinstance(low, (int, float, np.integer, np.float_))
    assert isinstance(high, (int, float, np.integer, np.float_))
    assert isinstance(pad, (int, float, np.integer, np.float_)) and 0.0 <= pad <= 1.0
    assert low <= high
    width: float = float(high) - float(low)
    pad: float = float(pad)
    return (low - width * pad, high + width * pad)


def rolling_avg(iterator: Union[Iterable, Iterator, Generator]) -> float:
    if not hasattr(iterator, "__iter__"):
        raise ValueError(
            f"Cannot calculate rolling average from an object which is neither an iterator or generator; "
            f"found object of type {type(iterator)}."
        )
    avg: float = 0
    for i, x in enumerate(iterator):
        avg = update_rolling_avg(avg, i, x)
    return avg


def update_rolling_avg(avg_i: float, i: int, val_i: float) -> float:
    """
    Calculates a rolling average.
    :param avg_i: the current average.
    :param i: the i'th index (starting from 0)
    :param val_i: the i'th value.
    :return: the updated average.

    Example usage:
    n: int = 1_000_000
    l: List[int] = list(range(1, n+1))  ## We know this adds up to n*(n+1)/2, thus the average is (n+1)/2
    avg: float = 0
    for i, x in enumerate(l):
        avg = update_rolling_avg(avg, i, x)
    assert avg == sum(l)/n == (n+1)/2
    """
    n: int = i + 1
    return ((n - 1) * avg_i + val_i) / n


def entropy(probabilities: np.ndarray) -> float:
    # Remove zero probabilities to avoid issues with logarithm
    if not isinstance(probabilities, np.ndarray):
        probabilities: np.ndarray = np.array(probabilities)
        assert is_numpy_float_array(probabilities)
    prob_sum: float = float(probabilities.sum())
    if abs(1 - prob_sum) > 1e-2:
        raise ValueError(f"Probabilities sum to {prob_sum}, should sum to 1")
    probabilities = probabilities[probabilities > 0]
    # probabilities += 1e-9
    _entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return _entropy


def relative_increase(
    prev: float,
    cur: float,
    *,
    how: Literal["ratio", "pct"] = "ratio",
    decimals: Optional[int] = None,
) -> float:
    assert how in {"ratio", "pct"}
    increase_frac: float = cur / prev
    if how == "ratio":
        if decimals is None:
            decimals: int = 5
        return round(increase_frac - 1, decimals)
    elif how == "pct":
        if decimals is None:
            decimals: int = 2
        return round(100 * (increase_frac - 1), decimals)
    elif how == "bps":
        if decimals is None:
            decimals: int = 1
        return round(100 * 100 * (increase_frac - 1), decimals)
    else:
        raise NotImplementedError(f'Unsupported `method`: "{how}"')


def to_pct(counts: pd.Series):  ## Converts value counts to percentages
    _sum = counts.sum()
    return pd.DataFrame(
        {
            "value": counts.index.tolist(),
            "count": counts.tolist(),
            "pct": counts.apply(lambda x: 100 * x / _sum).tolist(),
            "count_str": counts.apply(lambda x: f"{x} of {_sum}").tolist(),
        }
    )
