import ast
import functools
import inspect
import re
import sys
import types
from ast import literal_eval
from typing import *

from pydantic import BaseModel, Extra, root_validator

from ._utils import get_default


def fn_str(fn):
    return f"{get_fn_spec(fn).resolved_name}"


get_current_fn_name = lambda n=0: sys._getframe(
    n + 1
).f_code.co_name  ## Ref: https://stackoverflow.com/a/31615605


def is_function(fn: Any) -> bool:
    ## Ref: https://stackoverflow.com/a/69823452/4900327
    return isinstance(
        fn,
        (
            types.FunctionType,
            types.MethodType,
            types.BuiltinFunctionType,
            types.BuiltinMethodType,
            types.LambdaType,
            functools.partial,
        ),
    )


def call_str_to_params(
    call_str: str,
    callable_name_key: str = "name",
    max_len: int = 1024,
) -> Tuple[List, Dict]:
    """Creates params dict from a call string."""
    if len(call_str) > max_len:  ## To prevent this attack: https://stackoverflow.com/a/54763776/4900327
        raise ValueError(f"We cannot parse `call_str` beyond {max_len} chars; found {len(call_str)} chars")
    call_str: str = call_str.strip()
    if not (call_str.find("(") < call_str.find(")")):
        raise ValueError(
            f"`call_str` must have one opening paren, followed by one closing paren; "
            f'found: `call_str`="{call_str}"'
        )
    if not call_str.endswith(")"):
        raise ValueError(f'`call_str` must end with a closing paren; found: `call_str`="{call_str}"')
    name: str = call_str.split("(")[0]
    args: List = []
    kwargs: Dict = {callable_name_key: name}
    if call_str != f"{name}()":
        ## We have some params:
        params_str: str = call_str.replace(f"{name}(", "")
        assert params_str.endswith(")")
        params_str: str = params_str[:-1]
        for param_str in params_str.split(","):
            param_str: str = param_str.strip()
            if "=" not in param_str:
                ## Not an arg-value pair, instead just arg:
                args.append(literal_eval(param_str))
            elif len(param_str.split("=")) != 2:
                ## Cannot resolve arg-value pair:
                raise ValueError(f'Found invalid arg-value pair "{param_str}" in `call_str`="{call_str}"')
            else:
                k, v = param_str.split("=")
                ## No, this is not a security issue. Ref: https://stackoverflow.com/a/7689085/4900327
                if k == name:
                    raise ValueError(f'Argument name and callable name overlap: "{name}"')
                kwargs[k] = literal_eval(v)
    return args, kwargs


def params_to_call_str(callable_name: str, args: List, kwargs: Dict) -> str:
    sep: str = ", "
    stringified = []
    if len(args) > 0:
        stringified.append(sep.join(args))
    if len(kwargs) > 0:
        stringified.append(
            sep.join([f"{k}={v}" for k, v in sorted(list(kwargs.items()), key=lambda x: x[0])])
        )
    return f"{callable_name}({sep.join(stringified)})"


def wrap_fn_output(fn: Callable, wrapper_fn: Callable) -> Callable:
    """
    Ensures a function always returns objects of a particular class.
    :param fn: original function to invoke.
    :param wrapper_fn: wrapper which takes as input the original function output and returns a different value.
    :return: wrapped function object.
    """

    def do(*args, **kwargs):
        return wrapper_fn(fn(*args, **kwargs))

    return do


def parsed_fn_source(function) -> Tuple[str, str]:
    # Get the source code of the function
    # Parse the source code into an AST
    parsed_source = ast.parse(inspect.getsource(function))
    # The first element of the body should be the FunctionDef node for the function
    function_node: Any = parsed_source.body[0]
    # Extract the body of the FunctionDef node
    fn_source: str = ast.unparse(function_node)
    # Convert the body back to source code strings
    fn_body: str = "\n".join([ast.unparse(stmt) for stmt in function_node.body])
    return fn_source, fn_body


class FunctionSpec(BaseModel):
    name: str
    qualname: str
    resolved_name: str
    source: str
    source_body: str
    args: Tuple[str, ...]
    varargs_name: Optional[str]
    kwargs: Tuple[str, ...]
    varkwargs_name: Optional[str]
    default_args: Dict[str, Any]
    default_kwargs: Dict[str, Any]
    ignored_args: Tuple[str, ...] = ("self", "cls")

    class Config:
        ## Ref for Pydantic mutability: https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
        allow_mutation = False
        ## Ref for Extra.forbid: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        extra = Extra.forbid
        ## Ref for Pydantic private attributes: https://pydantic-docs.helpmanual.io/usage/models/#private-model-attributes
        underscore_attrs_are_private = True
        ## Validates default values. Ref: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        validate_all = True
        ## Validates typing by `isinstance` check. Ref: https://pydantic-docs.helpmanual.io/usage/model_config/#options
        arbitrary_types_allowed = True

    @root_validator(pre=False)
    def _remove_ignored(cls, params: Dict) -> Dict:
        ignored_args: Tuple[str, ...] = params["ignored_args"]
        params["args"] = tuple(arg_name for arg_name in params["args"] if arg_name not in ignored_args)
        params["kwargs"] = tuple(arg_name for arg_name in params["kwargs"] if arg_name not in ignored_args)
        params["default_args"] = dict(
            (arg_name, default_val)
            for arg_name, default_val in params["default_args"].items()
            if arg_name not in ignored_args
        )
        params["default_kwargs"] = dict(
            (arg_name, default_val)
            for arg_name, default_val in params["default_kwargs"].items()
            if arg_name not in ignored_args
        )
        return params

    @property
    def args_and_kwargs(self) -> Tuple[str, ...]:
        return self.args + self.kwargs

    @property
    def default_args_and_kwargs(self) -> Dict[str, Any]:
        return {**self.default_args, **self.default_kwargs}

    @property
    def required_args_and_kwargs(self) -> Tuple[str, ...]:
        default_args_and_kwargs: Dict[str, Any] = self.default_args_and_kwargs
        return tuple(arg_name for arg_name in self.args_and_kwargs if arg_name not in default_args_and_kwargs)

    @property
    def num_args(self) -> int:
        return len(self.args)

    @property
    def num_kwargs(self) -> int:
        return len(self.kwargs)

    @property
    def num_args_and_kwargs(self) -> int:
        return self.num_args + self.num_kwargs

    @property
    def num_default_args(self) -> int:
        return len(self.default_args)

    @property
    def num_default_kwargs(self) -> int:
        return len(self.default_kwargs)

    @property
    def num_default_args_and_kwargs(self) -> int:
        return self.num_default_args + self.num_default_kwargs

    @property
    def num_required_args_and_kwargs(self) -> int:
        return self.num_args_and_kwargs - self.num_default_args_and_kwargs


def get_fn_spec(fn: Callable) -> FunctionSpec:
    if hasattr(fn, "__wrapped__"):
        """
        if a function is wrapped with decorators, unwrap and get all args
        eg: pd.read_csv.__code__.co_varnames returns (args, kwargs, arguments) as its wrapped by a decorator @deprecate_nonkeyword_arguments
        This line ensures to unwrap all decorators recursively
        """
        return get_fn_spec(fn.__wrapped__)
    argspec: inspect.FullArgSpec = inspect.getfullargspec(fn)  ## Ref: https://stackoverflow.com/a/218709

    args: Tuple[str, ...] = tuple(get_default(argspec.args, []))
    varargs_name: Optional[str] = argspec.varargs

    kwargs: Tuple[str, ...] = tuple(get_default(argspec.kwonlyargs, []))
    varkwargs_name: Optional[str] = argspec.varkw

    default_args: Tuple[Any, ...] = get_default(argspec.defaults, tuple())
    default_args: Dict[str, Any] = dict(
        zip(
            argspec.args[-len(default_args) :],  ## Get's last len(default_args) values from the args list.
            default_args,
        )
    )
    default_kwargs: Dict[str, Any] = get_default(argspec.kwonlydefaults, dict())

    try:
        source, source_body = parsed_fn_source(fn)
    except IndentationError:
        source = inspect.getsource(fn)
        source_args_and_body = re.sub(r"^\s*(def\s+\w+\()", "", source, count=1, flags=re.MULTILINE).strip()
        source_body: str = source_args_and_body  ## Better than nothing.
    return FunctionSpec(
        name=fn.__name__,
        qualname=fn.__qualname__,
        resolved_name=fn.__module__ + "." + fn.__qualname__,
        source=source,
        source_body=source_body,
        args=args,
        varargs_name=varargs_name,
        kwargs=kwargs,
        varkwargs_name=varkwargs_name,
        default_args=default_args,
        default_kwargs=default_kwargs,
    )


def get_fn_args(
    fn: Union[Callable, FunctionSpec],
    *,
    ignore: Tuple[str, ...] = ("self", "cls", "kwargs"),
    include_args: bool = True,
    include_kwargs: bool = True,
    include_default: bool = True,
) -> Tuple[str, ...]:
    if isinstance(fn, FunctionSpec):
        fn_spec: FunctionSpec = fn
    else:
        fn_spec: FunctionSpec = get_fn_spec(fn)
    arg_names: List[str] = list()
    if include_args:
        arg_names.extend(fn_spec.args)
    if include_kwargs:
        arg_names.extend(fn_spec.kwargs)
    if include_default is False:
        ignore: List[str] = (
            list(ignore) + list(fn_spec.default_args.keys()) + list(fn_spec.default_kwargs.keys())
        )
    ignore: Set[str] = set(ignore)
    arg_names: Tuple[str, ...] = tuple(a for a in arg_names if a not in ignore)
    return arg_names


def filter_kwargs(fns: Union[Callable, List[Callable], Tuple[Callable, ...]], **kwargs) -> Dict[str, Any]:
    to_keep: Set = set()
    if isinstance(fns, (list, set, tuple)):
        fns = list(fns)
    else:
        fns = [fns]
    for fn in fns:
        fn_args: Tuple[str, ...] = get_fn_args(fn)
        to_keep.update(set(fn_args))
    filtered_kwargs: Dict[str, Any] = {k: kwargs[k] for k in kwargs if k in to_keep}
    return filtered_kwargs
