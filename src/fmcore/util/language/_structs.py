import random
import re
from ast import literal_eval
from collections import defaultdict
from contextlib import contextmanager
from typing import *

import numpy as np
import pandas as pd
from pandas.core.frame import Series as PandasSeries, DataFrame as PandasDataFrame
from pydantic.typing import Literal

from ._alias import set_param_from_alias
from ._autoenum import AutoEnum
from ._import import optional_dependency
from ._utils import get_default, is_not_null

ListOrTuple = Union[List, Tuple]
DataFrameOrSeries = Union[PandasSeries, PandasDataFrame]
SeriesOrArray1D = Union[PandasSeries, List, Tuple, np.ndarray]
DataFrameOrArray2D = Union[PandasSeries, PandasDataFrame, List, List[List], np.ndarray]
SeriesOrArray1DOrDataFrameOrArray2D = Union[SeriesOrArray1D, DataFrameOrArray2D]


def not_impl(
        param_name: str,
        param_val: Any,
        supported: Optional[Union[List, Set, Tuple, Any]] = None,
) -> Exception:
    if not isinstance(param_name, str):
        raise ValueError(f'First value `param_name` must be a string.')
    param_val_str: str = str(param_val)
    if len(param_val_str) > 100:
        param_val_str: str = '\n' + param_val_str
    if supported is not None:
        supported: List = as_list(supported)
        return NotImplementedError(
            f'Unsupported value for param `{param_name}`. Valid values are: {supported}; '
            f'found {type(param_val)} having value: {param_val_str}'
        )

    return NotImplementedError(
        f'Unsupported value for param `{param_name}`; '
        f'found {type(param_val)} having value: {param_val_str}'
    )


## ======================== List utils ======================== ##
def is_list_like(l: Any) -> bool:
    with optional_dependency('dask'):
        from dask.dataframe.core import Series as DaskSeries
        if isinstance(l, (list, tuple, ValuesView, ItemsView, PandasSeries, DaskSeries)):
            return True
    if isinstance(l, (list, tuple, ValuesView, ItemsView, PandasSeries)):
        return True
    if isinstance(l, np.ndarray) and l.ndim == 1:
        return True
    return False


def is_not_empty_list_like(l: ListOrTuple) -> bool:
    return is_list_like(l) and len(l) > 0


def is_empty_list_like(l: ListOrTuple) -> bool:
    return not is_not_empty_list_like(l)


def assert_not_empty_list(l: List):
    assert is_not_empty_list(l)


def assert_not_empty_list_like(l: ListOrTuple, error_message=''):
    assert is_not_empty_list_like(l), error_message


def is_not_empty_list(l: List) -> bool:
    return isinstance(l, list) and len(l) > 0


def is_empty_list(l: List) -> bool:
    return not is_not_empty_list(l)


def as_list(l) -> List:
    if is_list_or_set_like(l):
        return list(l)
    return [l]


def list_pop_inplace(l: List, *, pop_condition: Callable) -> List:
    assert isinstance(l, list)  ## Needs to be a mutable
    ## Iterate backwards to preserve indexes while iterating
    for i in range(len(l) - 1, -1, -1):  # Iterate backwards
        if pop_condition(l[i]):
            l.pop(i)  ## Remove the item inplace
    return l


def set_union(*args) -> Set:
    _union: Set = set()
    for s in args:
        if isinstance(s, (pd.Series, np.ndarray)):
            s: List = s.tolist()
        s: Set = set(s)
        _union: Set = _union.union(s)
    return _union


def set_intersection(*args) -> Set:
    _intersection: Optional[Set] = None
    for s in args:
        if isinstance(s, (pd.Series, np.ndarray)):
            s: List = s.tolist()
        s: Set = set(s)
        if _intersection is None:
            _intersection: Set = s
        else:
            _intersection: Set = _intersection.intersection(s)
    return _intersection


def filter_string_list(l: List[str], pattern: str, ignorecase: bool = False) -> List[str]:
    """
    Filter a list of strings based on an exact match to a regex pattern. Leaves non-string items untouched.
    :param l: list of strings
    :param pattern: Regex pattern used to match each item in list of strings.
    Strings which are not a regex pattern will be expected to exactly match.
    E.g. the pattern 'abcd' will only match the string 'abcd'.
    To match 'abcdef', pattern 'abcd.*' should be used.
    To match 'xyzabcd', patterm '.*abcd' should be used.
    To match 'abcdef', 'xyzabcd' and 'xyzabcdef', patterm '.*abcd.*' should be used.
    :param ignorecase: whether to ignore case while matching the pattern to the strings.
    :return: filtered list of strings which match the pattern.
    """
    if not pattern.startswith('^'):
        pattern = '^' + pattern
    if not pattern.endswith('$'):
        pattern = pattern + '$'
    flags = 0
    if ignorecase:
        flags = flags | re.IGNORECASE
    return [x for x in l if not isinstance(x, str) or len(re.findall(pattern, x, flags=flags)) > 0]


def keep_values(
        a: Union[List, Tuple, Set, Dict],
        values: Any,
) -> Union[List, Tuple, Set, Dict]:
    values: Set = as_set(values)
    if isinstance(a, list):
        return list(x for x in a if x in values)
    elif isinstance(a, tuple):
        return tuple(x for x in a if x in values)
    elif isinstance(a, set):
        return set(x for x in a if x in values)
    elif isinstance(a, dict):
        return {k: v for k, v in a.items() if v in values}
    raise NotImplementedError(f'Unsupported data structure: {type(a)}')


def remove_values(
        a: Union[List, Tuple, Set, Dict],
        values: Any,
) -> Union[List, Tuple, Set, Dict]:
    values: Set = as_set(values)
    if isinstance(a, list):
        return list(x for x in a if x not in values)
    elif isinstance(a, tuple):
        return tuple(x for x in a if x not in values)
    elif isinstance(a, set):
        return set(x for x in a if x not in values)
    elif isinstance(a, dict):
        return {k: v for k, v in a.items() if v not in values}
    raise NotImplementedError(f'Unsupported data structure: {type(a)}')


def remove_nulls(
        a: Union[List, Tuple, Set, Dict],
) -> Union[List, Tuple, Set, Dict]:
    if isinstance(a, list):
        return list(x for x in a if is_not_null(x))
    elif isinstance(a, tuple):
        return tuple(x for x in a if is_not_null(x))
    elif isinstance(a, set):
        return set(x for x in a if is_not_null(x))
    elif isinstance(a, dict):
        return {k: v for k, v in a.items() if is_not_null(v)}
    raise NotImplementedError(f'Unsupported data structure: {type(a)}')


def elvis(d: Optional[Union[Dict, Any]], *args) -> Optional[Any]:
    if len(args) == 0:
        raise ValueError('Must pass non-empty list of keys to match when using elvis operator')
    val: Union[Dict, Any] = get_default(d, {})
    for k in args:
        val: Union[Dict, Any] = get_default(val, {})
        if isinstance(val, dict):
            val: Union[Dict, Any] = val.get(k)
        else:
            return val
    return val


## ======================== Tuple utils ======================== ##
def as_tuple(l) -> Tuple:
    if is_list_or_set_like(l):
        return tuple(l)
    return (l,)


## ======================== Set utils ======================== ##
def is_set_like(l: Any) -> bool:
    return isinstance(l, (set, frozenset, KeysView))


def is_list_or_set_like(l: Union[List, Tuple, np.ndarray, PandasSeries, Set, frozenset]):
    return is_list_like(l) or is_set_like(l)


def get_subset(small_list: ListOrTuple, big_list: ListOrTuple) -> Set:
    assert is_list_like(small_list)
    assert is_list_like(big_list)
    return set.intersection(set(small_list), set(big_list))


def is_subset(small_list: ListOrTuple, big_list: ListOrTuple) -> bool:
    return len(get_subset(small_list, big_list)) == len(small_list)


def as_set(s) -> Set:
    if isinstance(s, set):
        return s
    if is_list_or_set_like(s):
        return set(s)
    return {s}


## ======================== Dict utils ======================== ##
def append_to_keys(d: Dict, prefix: Union[List[str], str] = '', suffix: Union[List[str], str] = '') -> Dict:
    if not is_dict_like(d):
        raise ValueError(f'Expected a dict-like object, found: {type(d)}')
    keys = set(d.keys())
    for k in keys:
        new_keys = {f'{p}{k}' for p in as_list(prefix)} \
                   | {f'{k}{s}' for s in as_list(suffix)} \
                   | {f'{p}{k}{s}' for p in as_list(prefix) for s in as_list(suffix)}
        for k_new in new_keys:
            d[k_new] = d[k]
    return d


def transform_keys_case(d: Dict, case: Literal['lower', 'upper'] = 'lower'):
    """
    Converts string dict keys to either uppercase or lowercase. Leaves non-string keys untouched.
    :param d: dict to transform
    :param case: desired case, either 'lower' or 'upper'
    :return: dict with case-transformed keys
    """
    if not is_dict_like(d):
        raise ValueError(f'Expected a dict-like object, found: {type(d)}')
    assert case in {'lower', 'upper'}
    out = {}
    for k, v in d.items():
        if isinstance(k, str):
            if case == 'lower':
                out[k.lower()] = v
            elif case == 'upper':
                out[k.upper()] = v
        else:
            out[k] = v
    return out


def transform_values_case(d: Dict, case: Literal['lower', 'upper'] = 'lower'):
    """
    Converts string dict values to either uppercase or lowercase. Leaves non-string values untouched.
    :param d: dict to transform
    :param case: desired case, either 'lower' or 'upper'
    :return: dict with case-transformed values
    """
    if not is_dict_like(d):
        raise ValueError(f'Expected a dict-like object, found: {type(d)}')
    assert case in {'lower', 'upper'}
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            if case == 'lower':
                out[k] = v.lower()
            elif case == 'upper':
                out[k] = v.upper()
        else:
            out[k] = v
    return out


def dict_set_default(d: Dict, default_params: Dict) -> Dict:
    """
    Sets default values in a dict for missing keys
    :param d: input dict
    :param default_params: dict of default values
    :return: input dict with default values populated for missing keys
    """
    if d is None:
        d = {}
    assert isinstance(d, dict)
    if default_params is None:
        return d
    assert isinstance(default_params, dict)
    for k, v in default_params.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            ## We need to go deeper:
            d[k] = dict_set_default(d[k], v)
        else:
            d.setdefault(k, v)
    return d


def sorted_dict(
        d: Dict,
        *,
        by: Literal['key', 'value'] = 'key',
        reverse: bool = False,
        order: Optional[List] = None,
) -> List[Tuple]:
    assert by in {'key', 'value'}
    if order is not None:
        order: List = as_list(order)
        assert by == 'key'
        out_d: Dict = {}
        for k in order:
            ## In order
            out_d[k] = d[k]
        for k in set(d.keys()) - set(order):
            ## Unordered
            out_d[k] = d[k]
        return list(out_d.items())
    else:
        if by == 'key':
            return sorted(d.items(), key=lambda x: str(x[0]), reverse=reverse)
        elif by == 'value':
            return sorted(d.items(), key=lambda x: str(x[1]), reverse=reverse)
        else:
            raise not_impl('by', by)


def dict_key_with_best_value(
        d: Dict,
        *,
        how: Literal['max', 'min'],
) -> Any:
    assert how in {'max', 'min'}
    sorted_items: List[Tuple] = sorted_dict(
        d,
        by='value',
        reverse={
            'min': False,
            'max': True,
        }[how]
    )
    return sorted_items[0][0]


def filter_keys(
        d: Dict,
        keys: Union[List, Tuple, Set, str],
        how: Literal['include', 'exclude'] = 'include',
) -> Dict:
    """
    Filter values in a dict based on a list of keys.
    :param d: dict to filter
    :param keys: list of keys to include/exclude.
    :param how: whether to keep or remove keys in filtered_keys list.
    :return: dict with filtered list of keys
    """
    if not is_dict_like(d):
        raise ValueError(f'Expected a dict-like object, found: {type(d)}')
    keys: Set = as_set(keys)
    if how == 'include':
        return keep_keys(d, keys)
    elif how == 'exclude':
        return remove_keys(d, keys)
    else:
        raise NotImplementedError(f'Invalid value for parameter `how`: "{how}"')


def filter_values(
        struct: Union[List, Tuple, Set, Dict, str],
        fn: Callable,
        *,
        raise_error: bool = True,
) -> Optional[Any]:
    if (is_list_like(struct) or is_set_like(struct)) and len(struct) > 0:
        return type(struct)([x for x in struct if fn(x)])
    elif is_dict_like(struct):
        return dict({k: v for k, v in struct.items() if fn(v)})
    if raise_error:
        raise ValueError(f'Unsupported structure: {type(struct)}')
    return None


def keep_keys(d: Dict, keys: Union[List, Tuple, Set, str]) -> Dict:
    keys: Set = as_set(keys)
    return {k: d[k] for k in keys if k in d}


def remove_keys(d: Dict, keys: Union[List, Tuple, Set, str]) -> Dict:
    keys: Set = as_set(keys)
    return {k: d[k] for k in d if k not in keys}


class UniqueDict(dict):
    def __setitem__(self, key, value):  ## Dict which rejects updates for existing keys.
        if key not in self:
            dict.__setitem__(self, key, value)
        else:
            raise KeyError("Key already exists")


def convert_and_filter_keys_on_enum(
        d: Dict,
        AutoEnumClass: AutoEnum.__class__,
        how: Literal['include', 'exclude'] = 'include',
) -> Dict:
    """
    Filter values in a dict based on those matching an enum.
    :param d: dict to filter.
    :param AutoEnumClass: AutoEnum class on which to filter.
    :param how: whether to keep or remove keys in the AutoEnum class.
    :return: dict with filtered list of keys
    """
    if not is_dict_like(d):
        raise ValueError(f'Expected a dict-like object, found: {type(d)}')
    if AutoEnumClass is None:
        return {}
    assert isinstance(AutoEnumClass, AutoEnum.__class__)
    d = AutoEnumClass.convert_keys(d)
    return filter_keys(d, list(AutoEnumClass), how=how)


def filter_keys_on_pattern(
        d: Dict,
        key_pattern: str,
        ignorecase: bool = False,
        how: Literal['include', 'exclude'] = 'include',
):
    """
    Filter string keys in a dict based on a regex pattern.
    :param d: dict to filter
    :param key_pattern: regex pattern used to match keys.
    :param how: whether to keep or remove keys.
    Follows same rules as `filter_string_list` method, i.e. only checks string keys and retains non-string keys.
    :return: dict with filtered keys
    """
    keys: List = list(d.keys())
    filtered_keys: List = filter_string_list(keys, key_pattern, ignorecase=ignorecase)
    return filter_keys(d, filtered_keys, how=how)


def is_not_empty_dict(d: Dict) -> bool:
    return is_dict_like(d) and len(d) > 0


def is_empty_dict(d: Dict) -> bool:
    return not is_not_empty_dict(d)


def assert_not_empty_dict(d: Dict):
    assert is_not_empty_dict(d)


def is_dict_like(d: Union[Dict, defaultdict]) -> bool:
    return isinstance(d, (dict, defaultdict))


def is_list_or_dict_like(d: Any) -> bool:
    return is_list_like(d) or is_dict_like(d)


def is_list_of_dict_like(d: List[Dict]) -> bool:
    if not is_list_like(d):
        return False
    for x in d:
        if not is_dict_like(x):
            return False
    return True


def is_dict_like_or_list_of_dict_like(d: Union[Dict, List[Dict]]) -> bool:
    if is_dict_like(d):
        return True
    elif is_list_like(d):
        return is_list_of_dict_like(d)
    return False


def eval_dict_values(params: Dict):
    if not isinstance(params, dict):
        raise ValueError(f"{params} should be of type dict")
    updated_dict = {}
    for parameter, value in params.items():
        try:
            updated_dict[parameter] = literal_eval(value)
        except:
            updated_dict[parameter] = value
    return updated_dict


def invert_dict(d: Dict) -> Dict:
    if not isinstance(d, dict):
        raise ValueError(f'{d} should be of type dict')
    d_inv: Dict = {v: k for k, v in d.items()}
    if len(d_inv) != len(d):
        raise ValueError(f'Dict is not invertible as values are not unique.')
    return d_inv


def iter_dict(d, depth: int = 1, *, _cur_depth: int = 0):
    """
    Recursively iterate over nested dictionaries and yield keys at each depth.

    :param d: The dictionary to iterate over.
    :param depth: The current depth of recursion (used for tracking depth of keys).
    :return: Yields tuples where the first elements are keys at different depths, and the last element is the value.
    """
    assert isinstance(d, dict), f'Input must be a dictionary, found: {type(d)}'
    assert isinstance(depth, int) and depth >= 1, f'depth must be an integer (1 or more)'

    for k, v in d.items():
        if isinstance(v, dict) and _cur_depth < depth - 1:
            # If the value is a dictionary, recurse
            for subkeys in iter_dict(v, _cur_depth=_cur_depth + 1, depth=depth):
                yield (k,) + subkeys
        else:
            # If the value is not a dictionary, yield the key-value pair
            yield (k, v)


## ======================== Utils for multiple collections ======================== ##
def only_item(
        d: Union[Dict, List, Tuple, Set, np.ndarray, PandasSeries],
        raise_error: bool = True,
) -> Union[Dict, List, Tuple, Set, np.ndarray, PandasSeries, Any]:
    if not (is_list_or_set_like(d) or is_dict_like(d)):
        return d
    if len(d) == 1:
        if is_dict_like(d):
            return next(iter(d.items()))
        return next(iter(d))
    if raise_error:
        raise ValueError(f'Expected input {type(d)} to have only one item; found {len(d)} elements.')
    return d


def only_key(d: Dict, raise_error: bool = True) -> Union[Any]:
    if not is_dict_like(d):
        return d
    if len(d) == 1:
        return next(iter(d.keys()))
    if raise_error:
        raise ValueError(f'Expected input {type(d)} to have only one item; found {len(d)} elements.')
    return d


def only_value(d: Dict, raise_error: bool = True) -> Union[Any]:
    if not is_dict_like(d):
        return d
    if len(d) == 1:
        return next(iter(d.values()))
    if raise_error:
        raise ValueError(f'Expected input {type(d)} to have only one item; found {len(d)} elements.')
    return d


def is_1d_array(l: Union[List, Tuple]):
    return is_list_like(l) and len(l) > 0 and not is_list_like(l[0])


def is_2d_array(l: Union[List, Tuple]):
    return is_list_like(l) and len(l) > 0 and is_list_like(l[0])


def convert_1d_or_2d_array_to_dataframe(data: SeriesOrArray1DOrDataFrameOrArray2D) -> PandasDataFrame:
    if is_1d_array(data):
        data: PandasSeries = convert_1d_array_to_series(data)
    if isinstance(data, PandasSeries) or is_2d_array(data):
        data: PandasDataFrame = pd.DataFrame(data)
    assert isinstance(data, PandasDataFrame)
    return data


def convert_1d_array_to_series(data: SeriesOrArray1D):
    if len(data) == 0:
        raise ValueError(f'Cannot convert empty data structure to series')
    if isinstance(data, PandasSeries):
        return data
    if not is_list_like(data):
        raise ValueError(f'Cannot convert non list-like data structure to series')
    return pd.Series(data)


def flatten1d(
        l: Union[List, Tuple, Set, Any],
        output_type: Type = list
) -> Union[List, Set, Tuple]:
    assert output_type in {list, set, tuple}
    if not is_list_or_set_like(l):
        return l
    out = []
    for x in l:
        out.extend(as_list(flatten1d(x)))
    return output_type(out)


def flatten2d(
        l: Union[List, Tuple, Set, Any],
        outer_type: Type = list,
        inner_type: Type = tuple,
) -> Union[List, Tuple, Set, Any]:
    assert outer_type in {list, set, tuple}
    assert inner_type in {list, set, tuple}
    if not is_list_or_set_like(l):
        return l
    out: List[Union[List, Set, Tuple]] = [
        flatten1d(x, output_type=inner_type)
        for x in l
    ]
    return outer_type(out)


def partial_sort(
        struct: Union[List[Any], Tuple[Any]],
        order: Union[List[Any], Tuple[Any], Any],
) -> Union[List[Any], Tuple[Any]]:
    """
    Partially sorts a list or tuple.
    """
    ## Dictionary to store the count of each element in order
    order: List[Any] = as_list(order)
    order_count: Dict[Any, int] = {item: 0 for item in order}

    # Two lists: one for elements in order and one for the rest
    ordered_part: List[Any] = []
    rest_part: List[Any] = []

    for item in struct:
        if item in order_count:
            # If the item is in order, increment the count and add to ordered_part
            order_count[item] += 1
        else:
            # Otherwise, add to rest_part
            rest_part.append(item)

    ## Construct the final ordered part based on the count
    for item in order:
        ordered_part.extend([item] * order_count[item])

    ## Combine the ordered part with the rest
    out: List[Any] = ordered_part + rest_part
    if isinstance(struct, tuple):
        return tuple(out)
    return out


def is_sorted(l: Union[List[Any], Tuple[Any, ...]], *, reverse: bool = False) -> bool:
    assert isinstance(l, (list, tuple))
    length = len(l)
    assert length > 0
    if length == 1:
        return True
    if reverse:
        l: List[Any] = list(l)[::-1]
    for x, x_next in zip(l[0:length - 1], l[1:length]):
        if x > x_next:
            return False
    return True


def get_unique(
        data: SeriesOrArray1DOrDataFrameOrArray2D,
        exclude_nans: bool = True
) -> Set[Any]:
    if data is None:
        return set()
    if isinstance(data, PandasSeries) or isinstance(data, PandasDataFrame):
        data: np.ndarray = data.values
    if is_2d_array(data):
        data: np.ndarray = convert_1d_or_2d_array_to_dataframe(data).values
    if not isinstance(data, np.ndarray):
        data: np.ndarray = np.array(data)
    flattened_data = data.ravel('K')  ## 1-D array of all data (w/ nans). Ref: https://stackoverflow.com/a/26977495
    if len(flattened_data) == 0:
        return set()
    if exclude_nans:
        flattened_data = flattened_data[~pd.isnull(flattened_data)]
    flattened_data = np.unique(flattened_data)
    return set(flattened_data)


def any_item(
        struct: Union[List, Tuple, Set, Dict, ValuesView, str],
        *,
        seed: Optional[int] = None,
        raise_error: bool = True,
) -> Optional[Any]:
    py_random: random.Random = random.Random(seed)
    if (is_list_like(struct) or is_set_like(struct)) and len(struct) > 0:
        return py_random.choice(tuple(struct))
    elif is_dict_like(struct):
        k: Any = any_key(struct, seed=seed, raise_error=raise_error)
        v: Any = struct[k]
        return k, v  ## Return an item
    elif isinstance(struct, str):
        return py_random.choice(struct)
    if raise_error:
        raise ValueError(f'Unsupported structure: {type(struct)}')
    return None


def any_key(d: Dict, *, seed: Optional[int] = None, raise_error: bool = True) -> Optional[Any]:
    py_random: random.Random = random.Random(seed)
    if is_not_empty_dict(d):
        return py_random.choice(sorted(list(d.keys())))
    if raise_error:
        raise ValueError(
            f'Expected input to be a non-empty dict; '
            f'found {type(d) if not is_dict_like(d) else "empty dict"}.'
        )
    return None


def any_value(d: Dict, *, seed: Optional[int] = None, raise_error: bool = True) -> Optional[Any]:
    k: Any = any_key(d, seed=seed, raise_error=raise_error)
    return d[k]


def first_item(
        struct: Union[List, Tuple, Set, Dict, str],
        *,
        raise_error: bool = True,
) -> Optional[Any]:
    if is_dict_like(struct):
        k: Any = first_key(struct, raise_error=raise_error)
        v: Any = struct[k]
        return k, v  ## Return an item
    elif is_list_like(struct) or is_set_like(struct) or isinstance(struct, str):
        return list(struct)[0]
    if raise_error:
        raise ValueError(f'Unsupported structure: {type(struct)}')
    return None


def first_key(d: Dict, *, raise_error: bool = True) -> Optional[Any]:
    if is_not_empty_dict(d):
        return list(d.keys())[0]
    if raise_error:
        raise ValueError(
            f'Expected input to be a non-empty dict; '
            f'found {type(d) if not is_dict_like(d) else "empty dict"}.'
        )
    return None


def first_value(d: Dict, *, raise_error: bool = True) -> Optional[Any]:
    k: Any = first_key(d, raise_error=raise_error)
    return d[k]


## ======================== Pandas utils ======================== ##
def get_num_non_null_columns_per_row(df: PandasDataFrame) -> PandasSeries:
    ## Ref: https://datascience.stackexchange.com/a/16801/35826
    assert isinstance(df, PandasDataFrame)
    return (~df.isna()).sum(axis=1)


def get_max_num_non_null_columns_per_row(df: PandasDataFrame) -> int:
    assert isinstance(df, PandasDataFrame)
    return get_num_non_null_columns_per_row(df).max()


@contextmanager
def pd_display(**kwargs):
    """
    Use pd.describe_option('display') to see all options.
    """
    try:
        from IPython.display import display
    except ImportError:
        display = print
    set_param_from_alias(params=kwargs, param='max_rows', alias=['num_rows', 'nrows', 'rows'], default=None)
    set_param_from_alias(params=kwargs, param='max_cols', alias=['num_cols', 'ncols', 'cols'], default=None)
    set_param_from_alias(params=kwargs, param='max_colwidth', alias=[
        'max_col_width',
        'max_columnwidth', 'max_column_width',
        'columnwidth', 'column_width',
        'colwidth', 'col_width',
    ], default=None)
    set_param_from_alias(params=kwargs, param='vertical_align', alias=['valign'], default='top')
    set_param_from_alias(params=kwargs, param='text_align', alias=['textalign'], default='left')
    set_param_from_alias(params=kwargs, param='ignore_css', alias=['css'], default=False)

    max_rows: Optional[int] = kwargs.get('max_rows')
    max_cols: Optional[int] = kwargs.get('max_cols')
    max_colwidth: Optional[int] = kwargs.get('max_colwidth')
    vertical_align: str = kwargs['vertical_align']
    text_align: str = kwargs['text_align']
    ignore_css: bool = kwargs['ignore_css']

    # print(kwargs)

    def disp(df: pd.DataFrame):
        css = [
            ## Align header to center
            {
                'selector': 'th',
                'props': [
                    ('vertical-align', 'center'),
                    ('text-align', 'center'),
                    ('padding', '10px'),
                ]
            },
            ## Align cell to top and left/center
            {
                'selector': 'td',
                'props': [
                    ('vertical-align', vertical_align),
                    ('text-align', text_align),
                    ('padding', '10px'),
                ]
            },

        ]
        if not ignore_css and isinstance(df, pd.DataFrame):
            df = df.style.set_table_styles(css)
        display(df)

    with pd.option_context(
            'display.max_rows', max_rows,
            'display.max_columns', max_cols,
            'max_colwidth', max_colwidth,
            'display.expand_frame_repr', False,
    ):
        yield disp


def pd_partial_column_order(df: pd.DataFrame, columns: List) -> pd.DataFrame:
    columns: List = as_list(columns)
    df_columns: List = list(df.columns)
    final_columns: List = []
    for col in columns:
        if col not in df_columns:
            raise ValueError(f'Column "{col}" not found in current {pd.DataFrame} columns: {df.columns}')
        final_columns.append(col)
    for col in df_columns:  ## Add all the remaining columns
        if col not in final_columns:
            final_columns.append(col)
    assert set(final_columns) == set(df_columns)
    return df[final_columns]


## ======================== NumPy utils ======================== ##
def is_numpy_integer_array(data: Any) -> bool:
    if not isinstance(data, np.ndarray):
        return False
    return issubclass(data.dtype.type, np.integer)


def is_numpy_float_array(data: Any) -> bool:
    if not isinstance(data, np.ndarray):
        return False
    return issubclass(data.dtype.type, float)


def is_numpy_string_array(data: Any) -> bool:
    if not isinstance(data, np.ndarray):
        return False
    return issubclass(data.dtype.type, str)


## Ref (from Pytorch tests):
## github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349
NUMPY_TO_TORCH_DTYPE_MAP = {}
with optional_dependency('torch'):
    import torch

    NUMPY_TO_TORCH_DTYPE_MAP = {
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.complex64: torch.complex64,
        np.complex128: torch.complex128
    }
    TORCH_TO_NUMPY_DTYPE_MAP = {v: k for k, v in NUMPY_TO_TORCH_DTYPE_MAP.items()}
