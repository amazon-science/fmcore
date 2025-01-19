import math
import random
from typing import *

import numpy as np
import pandas as pd
from pydantic import confloat, conint
from pydantic.typing import Literal

from ._import import optional_dependency
from ._structs import as_list, as_set, flatten1d, is_dict_like, is_list_like, is_set_like, is_sorted
from ._typing import type_str
from ._utils import get_default, is_null, is_scalar

FractionalBool = Union[confloat(ge=0.0, le=1.0), bool]
SampleSizeType = Union[confloat(gt=0.0, le=1.0), conint(gt=1)]


def resolve_fractional_bool(fractional_bool: Optional[FractionalBool], seed: int = None) -> bool:
    if fractional_bool in {0.0, False, None}:
        return False
    elif fractional_bool in {1.0, False, True}:
        return True
    else:
        rnd: float = np.random.RandomState(seed=seed).random()
        return rnd <= fractional_bool


def resolve_sample_size(sample_size: Optional[SampleSizeType], length: int) -> conint(ge=0):
    if sample_size in {1.0, True}:
        n = length
    elif 0.0 < sample_size < 1.0:
        n: int = math.ceil(sample_size * length)  ## Use at least 1 row.
    elif isinstance(sample_size, int) and 1 < sample_size:
        n: int = sample_size
    else:
        raise ValueError(f"Invalid value for `sample_size`: {sample_size}")
    n: int = min(n, length)
    return n


def infer_np_dtype(
    data: Union[List, np.ndarray, pd.Series, "torch.Tensor"],
    sample_size: SampleSizeType = True,
    str_to_object: bool = True,
    return_str_for_collection: bool = False,
) -> Optional[Union[np.dtype, Type, str]]:
    """
    Fast inference of the numpy dtype in a list.
    Note: we cannot use pandas.api.types.infer_dtype because it returns Pandas dtypes, not numpy.

    :param data: data collection (usually a list or tuple).
    :param sample_size: amount of data to subsample (without replacement) in order to determine the dtype.
        If False, it will not subsample data. If True, it will use entire data.
        If 0.0 < sample < 1.0, then we will subsample a fraction of the data.
        If 1 <= sample, we will subsample these many rows of data.
    :param str_to_object: whether to treat string as objects rather than np.unicode_ (like "U<1").
    :param return_str_for_collection: whether to return the string 'collection' for collections like list, set,
        numpy array, etc.
    :return:
    """
    if isinstance(data, (np.ndarray, pd.Series)):
        return data.dtype
    with optional_dependency("torch"):
        import torch

        from ._structs import TORCH_TO_NUMPY_DTYPE_MAP

        if isinstance(data, torch.Tensor):
            return TORCH_TO_NUMPY_DTYPE_MAP[data.dtype]

    data: List = as_list(data)
    dtypes: Set[Union[Type, np.dtype]] = set()
    has_nulls: bool = False
    for x in random_sample(data, n=sample_size, replacement=False):
        if str_to_object and np.issubdtype(type(x), np.character):
            ## Fast convert str, np.str_ and np.unicode_ to object:
            return object
        if not is_scalar(x):
            ## Fast return for collections such as list, tuple, dict, set, np.ndarray, Tensors.
            if return_str_for_collection:
                return "collection"
            return object
        if is_null(x):  ## Checks NaNs, None, and pd.NaT
            has_nulls: bool = True
        else:
            dtypes.add(type(x))
    if len(dtypes) == 0:
        ## All NaNs / None
        return None
    elif len(dtypes) == 1:
        dtype = next(iter(dtypes))
        ## Ref: https://numpy.org/doc/stable/reference/arrays.dtypes.html#Built-in%20Python%20types
        if dtype in {bool, np.bool_, float, np.float_, complex, np.complex_, bytes}:
            return np.dtype(dtype)
    return _np_dtype_fallback(dtypes, has_nulls=has_nulls, str_to_object=str_to_object)


def _np_dtype_fallback(dtypes: Union[Type, Set[Type]], has_nulls: bool, str_to_object: bool):
    ## We have one or more dtypes, which might be Python types or Numpy dtypes.
    ## We will now check if all the dtypes have a common parent, based on the NumPy scalar types hierarchy:
    ## i.e. https://numpy.org/doc/stable/reference/arrays.scalars.html
    if all_are_np_subtypes(
        dtypes,
        {
            np.bool_,
        },
    ):
        if has_nulls:
            return np.float_  ## Converts None to NaN, and True/False to 1.0/0.0
        return np.bool_
    elif all_are_np_subtypes(dtypes, {np.bool_, np.integer}):
        if has_nulls:
            return np.float_  ## Converts None to NaN, True/False to 1.0/0.0, and 123 to 123.0
        return np.int_
    elif all_are_np_subtypes(dtypes, {np.bool_, np.integer, np.floating}):
        return np.float_
    elif all_are_np_subtypes(
        dtypes,
        {
            np.character,
        },
    ):
        if str_to_object:
            return object
        return np.unicode_
    elif all_are_np_subtypes(dtypes, {np.bool_, np.integer, np.floating, np.complex_}):
        return np.complex_
    ## Multiple, heterogeneous and incompatible types, return as object
    return object


def all_are_np_subtypes(
    dtypes: Union[Type, Set[Type]],
    parent_dtypes: Union[Type, Set[Type]],
) -> bool:
    ## Note: the following hold for Python types when checking with np.issubdtype:
    ## np.issubdtype(bool, np.bool_) is True
    ## np.issubdtype(int, np.integer) is True (however, np.issubdtype(bool, np.integer) is False)
    ## np.issubdtype(float, np.floating) is True (however, np.issubdtype(int, np.floating) is False)
    ## np.issubdtype(complex, np.complex_) is True (however, np.issubdtype(float, np.complex_) is False)
    ## np.issubdtype(str, np.character) is True
    dtypes: Set[Type] = as_set(dtypes)
    parent_dtypes: Set[Type] = as_set(parent_dtypes)
    return all(
        {any({np.issubdtype(dtype, parent_dtype) for parent_dtype in parent_dtypes}) for dtype in dtypes}
    )


def random_sample(
    data: Union[List, Tuple, Set, np.ndarray],
    n: SampleSizeType,
    *,
    replacement: bool = False,
    seed: Optional[int] = None,
) -> Union[List, np.ndarray]:
    """
    Sample data randomly from a list or numpy array, with or without replacement.
    :param data: list or numpy array to randomly subsample.
    :param n: size of the sample to return.
    :param replacement: whether to sample with replacement or not.
    :param seed: optional random seed to use for reproducibility.
    :return: list or numpy array of randomly-sampled data.
    """
    np_random = np.random.RandomState(seed)
    py_random = random.Random(seed)
    if is_set_like(data):
        data: List = list(data)
    if not is_list_like(data):
        raise ValueError(
            f"Input `data` must be {list}, {tuple} or {np.ndarray}; found object of type {type(data)}"
        )
    if len(data) == 1:
        return data
    l: Union[List, np.ndarray] = data
    length: int = len(l)
    n: int = resolve_sample_size(sample_size=n, length=length)
    if replacement:
        ## Subsample with replacement:
        ## Ref: https://stackoverflow.com/a/71892814/4900327
        if isinstance(l, (list, tuple)):
            if n < 50:
                return py_random.choices(l, k=n)
            else:
                return [l[idx] for idx in np_random.randint(0, len(l), n)]
        elif isinstance(l, np.ndarray):
            if n < 25:
                return [l[idx] for idx in (py_random.randrange(0, len(l)) for _ in range(n))]
            else:
                return np_random.choice(l, n, replace=True)
    else:
        ## Subsample without replacement:
        ## Ref: https://stackoverflow.com/a/71892814/4900327
        if isinstance(l, (list, tuple)):
            return py_random.sample(l, n)
        elif isinstance(l, np.ndarray):
            return np_random.choice(l, n, replace=False)
    raise NotImplementedError(f"Unsupported input data type: {type(data)}")


def values_dist(vals: Union[List, Tuple, np.ndarray, pd.Series]) -> pd.Series:
    assert isinstance(vals, (list, tuple, np.ndarray, pd.Series))
    val_counts: pd.Series = pd.Series(Counter(vals))  ## Includes nan and None as keys.
    return val_counts / val_counts.sum()


def sample_idxs_match_distribution(
    source: Union[List, Tuple, np.ndarray, pd.Series],
    target: Union[List, Tuple, np.ndarray, pd.Series],
    n: Optional[int] = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
    target_is_dist: bool = False,
) -> np.ndarray:
    """
    Values from current series based on another distribution, and return randomly-shuffled indexes from the source.
    Selecting these indexes will give a distribution from the source whicha matches that of the target distribution.
    """
    if not target_is_dist:
        target_prob_dist: pd.Series = values_dist(target)
    else:
        target_prob_dist: pd.Series = target
    assert isinstance(target_prob_dist, pd.Series)
    assert (
        abs(float(target_prob_dist.sum()) - 1.0) <= 1e-2
    )  ## Sum of probs should be exactly or very close to 1.

    assert isinstance(source, (list, tuple, np.ndarray, pd.Series))
    source_vc: pd.Series = pd.Series(Counter(source))
    # print(f'\nsource_vc:\n{source_vc}')
    # print(f'\ntarget_prob_dist:\n{target_prob_dist}')
    missing_source_vals: Set = set(target_prob_dist.index) - set(source_vc.index)
    if len(missing_source_vals) > 0:
        raise ValueError(
            f"Cannot sample; the following values are missing in the source: {missing_source_vals}"
        )

    n: int = get_default(n, len(source))
    max_n_sample: pd.Series = (source_vc / target_prob_dist).apply(
        lambda max_n_sample_category: min(max_n_sample_category, n),
    )
    # print(f'\n\nmax_n_sample:\n{max_n_sample}')
    max_n_sample: int = math.floor(min(max_n_sample.dropna()))
    # print(f'Max possible sample size: {max_n_sample}')
    source_value_wise_count_to_sample: pd.Series = (target_prob_dist * max_n_sample).round(0).astype(int)
    source_value_wise_count_to_sample: Dict[Any, int] = source_value_wise_count_to_sample.to_dict()
    ## Select random indexes:
    source_val_idxs: Dict[Any, List[int]] = {val: [] for val in source_vc.index}
    for idx, val in enumerate(source):
        if val in source_value_wise_count_to_sample:
            source_val_idxs[val].append(idx)
    sampled_idxs: np.array = np.array(
        flatten1d(
            [
                random_sample(source_val_idxs[val], n=req_source_val_count, seed=seed)
                for val, req_source_val_count in source_value_wise_count_to_sample.items()
            ]
        )
    )
    if shuffle:
        sampled_idxs: np.ndarray = np.random.RandomState(seed).permutation(sampled_idxs)
    return sampled_idxs


def random_cartesian_product(*lists, seed: Optional[int] = None, n: int):
    rnd = random.Random(seed)
    cartesian_idxs: Set[Tuple[int, ...]] = set()
    list_lens: List[int] = [len(l) for l in lists]
    max_count: int = 1
    for l_len in list_lens:
        max_count *= l_len
    if max_count < n:
        raise ValueError(f"At most {max_count} cartesian product elements can be created.")
    while len(cartesian_idxs) < n:
        rnd_idx: Tuple[int, ...] = tuple(rnd.randint(0, l_len - 1) for l_len in list_lens)
        if rnd_idx not in cartesian_idxs:
            cartesian_idxs.add(rnd_idx)
            elem = []
            for l_idx, l in zip(rnd_idx, lists):
                elem.append(l[l_idx])
            yield elem


def argmax(d: Union[List, Tuple, np.ndarray, Dict, Set]) -> Any:
    if is_set_like(d):
        raise ValueError(f"Cannot get argmax from a {type_str(d)}.")
    if is_dict_like(d):
        ## Get key pertaining to max value:
        return max(d, key=d.get)
    assert is_list_like(d)
    return max([(i, x) for (i, x) in enumerate(d)], key=lambda x: x[1])[0]


def argmin(d: Union[List, Tuple, np.ndarray, Dict, Set]) -> Any:
    if is_set_like(d):
        raise ValueError(f"Cannot get argmin from a {type_str(d)}.")
    if is_dict_like(d):
        ## Get key pertaining to max value:
        return min(d, key=d.get)
    assert is_list_like(d)
    return min([(i, x) for (i, x) in enumerate(d)], key=lambda x: x[1])[0]


def best_k(
    vals: np.ndarray,
    k: int,
    *,
    how: Literal["min", "max"],
    sort: Optional[Literal["ascending", "descending"]] = None,
    indexes_only: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Efficiently gets the top-k elements from a numpy array."""
    assert isinstance(k, int) and k > 0
    ## np.argpartition creates a new array with the top-k/bottom-k scores in the head/tail k elements,
    ## but these k are not actually sorted.
    if how == "min":
        sort: str = sort if sort is not None else "ascending"
        bottom_k_idxs: np.ndarray = np.argpartition(vals, k, axis=0)[:k]
        ## Index vals to get bottom-k values, unsorted:
        bottom_k_vals: np.ndarray = vals[bottom_k_idxs]
        ## Get argsorted indexes for the bottom-k values (between 1 & k).
        ## We then use this to index the bottom-k-indexes array:
        if sort == "ascending":
            bottom_k_idxs_sorted: np.ndarray = bottom_k_idxs[bottom_k_vals.argsort(axis=0)]
            bottom_k_vals_sorted = np.sort(bottom_k_vals, axis=0)
        elif sort == "descending":
            bottom_k_idxs_sorted: np.ndarray = bottom_k_idxs[bottom_k_vals.argsort(axis=0)[::-1]]
            bottom_k_vals_sorted = np.sort(bottom_k_vals, axis=0)[::-1]
        else:
            raise NotImplementedError(f"Unsupported value of `sort`: {sort}")
        # print(f'bottom_k_vals_sorted: {bottom_k_vals_sorted}')
        # print(f'bottom_k_idxs_sorted: {bottom_k_idxs_sorted}')
        # assert bool((vals[bottom_k_idxs_sorted] == bottom_k_vals_sorted).all())
        if indexes_only:
            return bottom_k_idxs_sorted
        return bottom_k_idxs_sorted, bottom_k_vals_sorted
    elif how == "max":
        sort: str = sort if sort is not None else "descending"
        top_k_idxs: np.ndarray = np.argpartition(vals, -k, axis=0)[-k:]
        ## Index vals to get top-k values, unsorted:
        top_k_vals: np.ndarray = vals[top_k_idxs]
        ## Get argsorted indexes for the top-k values (between 1 & k).
        ## We then use this to index the top-k-indexes array:
        if sort == "ascending":
            top_k_idxs_sorted: np.ndarray = top_k_idxs[top_k_vals.argsort(axis=0)]
            top_k_vals_sorted = np.sort(top_k_vals, axis=0)
        elif sort == "descending":
            top_k_idxs_sorted: np.ndarray = top_k_idxs[top_k_vals.argsort(axis=0)[::-1]]
            top_k_vals_sorted = np.sort(top_k_vals, axis=0)[::-1]
        else:
            raise NotImplementedError(f"Unsupported value of `sort`: {sort}")
        # print(f'top_k_vals_sorted: {top_k_vals_sorted}')
        # print(f'top_k_idxs_sorted: {top_k_idxs_sorted}')
        # assert bool((vals[top_k_idxs_sorted] == top_k_vals_sorted).all())
        if indexes_only:
            return top_k_idxs_sorted
        return top_k_idxs_sorted, top_k_vals_sorted
    else:
        raise ValueError(f"Unsupported value for `how`: {how}")


def shuffle_items(
    struct: Union[List, Tuple, Set, Dict, str],
    *,
    seed: Optional[int] = None,
    dict_return_values: bool = False,
) -> Generator[Any, None, None]:
    if is_set_like(struct):
        struct: Tuple = tuple(struct)
    elif is_dict_like(struct):
        if dict_return_values:
            struct: Tuple = tuple(struct.values())
        else:
            struct: Tuple = tuple(struct.items())
    rnd_idxs: List[int] = list(range(len(struct)))
    random.Random(seed).shuffle(rnd_idxs)
    for rnd_idx in rnd_idxs:
        yield struct[rnd_idx]


_Comparable = Union[int, float, str]


def binary_search(
    l: Union[List[_Comparable], Tuple[_Comparable, ...]],
    target: _Comparable,
    *,
    return_tuple: bool = False,
) -> Union[Tuple[Optional[_Comparable], Optional[_Comparable]], _Comparable]:
    if not is_sorted(l):
        l: List[_Comparable] = sorted(l)
    low: int = 0
    high: int = len(l) - 1
    while low <= high:
        mid = (low + high) // 2
        if l[mid] == target:
            if return_tuple:
                return l[mid], l[mid]
            return l[mid]
        elif l[mid] < target:
            low: int = mid + 1
        else:
            high: int = mid - 1

    ## When the target is not found, set lower and upper bounds
    lower: _Comparable = l[high] if high >= 0 else None
    upper: _Comparable = l[low] if low < len(l) else None

    return lower, upper
