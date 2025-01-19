import functools
import inspect
import json
import typing
from abc import ABC
from typing import *

import numpy as np
import typing_extensions
from pydantic import (
    BaseModel,
    Extra,
    Field,
    constr,
    create_model_from_typeddict,
    root_validator,
    validate_arguments,
)
from pydantic.fields import Undefined

from ._autoenum import AutoEnum
from ._function import call_str_to_params, get_fn_spec, is_function, params_to_call_str
from ._string import NeverFailJsonEncoder, String
from ._structs import as_list, as_set, is_list_like
from ._utils import get_default


def type_str(data: Any) -> str:
    if isinstance(data, type):
        if issubclass(data, Parameters):
            out: str = data.class_name
        else:
            out: str = str(data.__name__)
    else:
        out: str = str(type(data))
    ## Crocodile brackets mess up Aim's logging, they are treated as HTML tags.
    out: str = out.replace("<", "").replace(">", "")
    return out


def is_abstract(Class: Type) -> bool:
    return ABC in Class.__bases__


## Ref: https://stackoverflow.com/a/13624858/4900327
class classproperty(property):
    def __get__(self, obj, objtype=None):
        return super(classproperty, self).__get__(objtype)

    def __set__(self, obj, value):
        super(classproperty, self).__set__(type(obj), value)

    def __delete__(self, obj):
        super(classproperty, self).__delete__(type(obj))


def safe_validate_arguments(f):
    names_to_fix = {n for n in BaseModel.__dict__ if not n.startswith("_")}

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        kwargs = {n[:-1] if n[:-1] in names_to_fix else n: v for n, v in kwargs.items()}
        return f(*args, **kwargs)

    def _create_param(p: inspect.Parameter) -> inspect.Parameter:
        default = Undefined if p.default is inspect.Parameter.empty else p.default
        return p.replace(name=f"{p.name}_", default=Field(default, alias=p.name))

    sig = inspect.signature(f)
    sig = sig.replace(
        parameters=[_create_param(p) if n in names_to_fix else p for n, p in sig.parameters.items()]
    )

    wrapper.__signature__ = sig
    wrapper.__annotations__ = {f"{n}_" if n in names_to_fix else n: v for n, v in f.__annotations__.items()}

    try:
        return validate_arguments(
            wrapper,
            config={
                "allow_population_by_field_name": True,
                "arbitrary_types_allowed": True,
            },
        )
    except Exception as e:
        raise ValueError(
            f"Error creating model for function {get_fn_spec(f).resolved_name}."
            f"\nEncountered Exception: {String.format_exception_msg(e)}"
        )


def check_isinstance(
    x: Optional[Any], y: Union[List[Type], Tuple[Type, ...], Type], raise_error: bool = True
):
    if x is None and y is type(None):
        return True
    assert isinstance(y, type) or (isinstance(y, (list, tuple)) and np.all([isinstance(z, type) for z in y]))
    if (isinstance(y, type) and isinstance(x, y)) or (
        isinstance(y, list) and np.any([isinstance(x, z) for z in y])
    ):
        return True
    if raise_error:
        y_str: str = ", ".join([type_str(_y) for _y in as_list(y)])
        raise TypeError(
            f"Input parameter must be of type `{y_str}`; found type `{type_str(x)}` with value:\n{x}"
        )
    return False


def check_isinstance_or_none(x: Optional[Any], y: Type, raise_error: bool = True):
    if x is None:
        return True
    return check_isinstance(x, y, raise_error=raise_error)


def check_issubclass_or_none(x: Optional[Any], y: Type, raise_error: bool = True):
    if x is None:
        return True
    return check_issubclass(x, y, raise_error=raise_error)


def check_issubclass(x: Optional[Any], y: Type, raise_error: bool = True):
    if x is None:
        return False
    assert isinstance(x, type)
    assert isinstance(y, type) or (isinstance(y, list) and np.all([isinstance(z, type) for z in y]))
    if (isinstance(y, type) and issubclass(x, y)) or (
        isinstance(y, list) and np.any([issubclass(x, z) for z in y])
    ):
        return True
    if raise_error:
        raise TypeError(
            f"Input parameter must be a subclass of type {str(y)}; found type {type(x)} with value {x}"
        )
    return False


def get_classvars(cls) -> List[str]:
    return [
        var_name
        for var_name, typing_ in typing.get_type_hints(cls).items()
        if typing_.__origin__ is typing.ClassVar
    ]


def get_classvars_typing(cls) -> Dict[str, Any]:
    return {
        var_name: typing_.__args__[0]
        for var_name, typing_ in typing.get_type_hints(cls).items()
        if typing.get_origin(typing_) is typing.ClassVar
    }


class Registry(ABC):
    """
    A registry for subclasses. When a base class extends Registry, its subclasses will automatically be registered,
     without any code in the base class to do so explicitly.
    This coding trick allows us to maintain the Dependency Inversion Principle, as the base class does not have to
     depend on any subclass implementation; in the base class code, we can instead retrieve the subclass in the registry
     using a key, and then interact with the retrieved subclass using the base class interface methods (which we assume
     the subclass has implemented as per the Liskov Substitution Principle).

    Illustrative example:
        Suppose we have abstract base class AbstractAnimal.
        This is registered as a registry via:
            class AbstractAnimal(Parameters, Registry, ABC):
                pass
        Then, subclasses of AbstractAnimal will be automatically registered:
            class Dog(AbstractAnimal):
                name: str
        Now, we can extract the subclass using the registered keys (of which the class-name is always included):
            AbstractAnimalSubclass = AbstractAnimal.get_subclass('Dog')
            dog = AbstractAnimalSubclass(name='Sparky')

        We can also set additional keys to register the subclass against:
            class AnimalType(AutoEnum):
                CAT = auto()
                DOG = auto()
                BIRD = auto()

            class Dog(AbstractAnimal):
                aliases = [AnimalType.DOG]

            AbstractAnimalSubclass = AbstractAnimal.get_subclass(AnimalType.DOG)
            dog = AbstractAnimalSubclass(name='Sparky')

        Alternately, the registry keys can be set using the _registry_keys() classmethod:
            class Dog(AbstractAnimal):
                @classmethod
                def _registry_keys(cls) -> List[Any]:
                    return [AnimalType.DOG]
    """

    _registry: ClassVar[Dict[Any, Dict[str, Type]]] = {}  ## Dict[key, Dict[classname, Class]
    _registry_base_class: ClassVar[Optional[Type[BaseModel]]] = None
    _classvars_typing_dict: ClassVar[Optional[Dict[str, Any]]] = None
    _classvars_BaseModel: ClassVar[Optional[Type[BaseModel]]] = None
    _allow_multiple_subclasses: ClassVar[bool] = False
    _allow_subclass_override: ClassVar[bool] = False
    _dont_register: ClassVar[bool] = False
    aliases: ClassVar[Tuple[str, ...]] = tuple()

    def __init_subclass__(cls, **kwargs):
        """
        Register any subclass with the base class. A child class is registered as long as it is imported/defined.
        """
        super().__init_subclass__(**kwargs)
        if cls in Registry.__subclasses__():
            ## The current class is a direct subclass of Registry (i.e. it is the base class of the hierarchy).
            cls._registry: Dict[Any, Dict[str, Type]] = {}
            cls._registry_base_class: Type = cls
            cls.__set_classvars_typing()
        else:
            ## The current class is a subclass of a Registry-subclass, and is not abstract; register this.
            if not is_abstract(cls) and not cls._dont_register:
                cls._pre_registration_hook()
                cls.__set_classvars_typing()
                cls.__validate_classvars_BaseModel()
                cls.__register_subclass()

    @classmethod
    def __set_classvars_typing(cls):
        classvars_typing_dict: Dict[str, Any] = {
            var_name: typing_
            for var_name, typing_ in get_classvars_typing(cls).items()
            if not var_name.startswith("_")
        }
        cls._classvars_typing_dict: ClassVar[Dict[str, Any]] = classvars_typing_dict

        class Config(Parameters.Config):
            extra = Extra.ignore

        cls._classvars_BaseModel: ClassVar[Type[BaseModel]] = create_model_from_typeddict(
            typing_extensions.TypedDict(f"{cls.__name__}_ClassVarsBaseModel", classvars_typing_dict),
            warnings=False,
            __config__=Config,
        )

    @classmethod
    def __validate_classvars_BaseModel(cls):
        ## Gives the impression of validating ClassVars on concrete subclasses in the hierarchy.
        classvar_values: Dict[str, Any] = {}
        for classvar, type_ in cls._classvars_typing_dict.items():
            if not hasattr(cls, classvar):
                if ABC not in cls.__bases__:
                    ## Any concrete class must have all classvars set with values.
                    raise ValueError(
                        f'You must set a value for class variable "{classvar}" on subclass "{cls.__name__}".\n'
                        f'Custom type-hints might be one reason why "{classvar}" is not recognized. '
                        f'If you have added custom type-hints, please try removing them and set "{classvar}" like so: `{classvar} = <value>`'
                    )
            else:
                classvar_value = getattr(cls, classvar)
                if hasattr(type_, "__origin__"):
                    if (
                        type_.__origin__ == typing.Union
                        and len(type_.__args__) == 2
                        and type(None) in type_.__args__
                    ):
                        ## It is something like Optional[str], Optional[List[str]], etc.
                        args = set(type_.__args__)
                        args.remove(type(None))
                        classvar_type = next(iter(args))
                    else:
                        classvar_type = type_.__origin__
                    if classvar_type in {set, list, tuple} and classvar_value is not None:
                        classvar_value = classvar_type(as_list(classvar_value))
                classvar_values[classvar] = classvar_value
        classvar_values: BaseModel = cls._classvars_BaseModel(**classvar_values)
        for classvar, type_ in cls._classvars_typing_dict.items():
            if not hasattr(cls, classvar):
                if ABC not in cls.__bases__:
                    ## Any concrete class must have all classvars set with values.
                    raise ValueError(
                        f'You must set a value for class variable "{classvar}" on subclass "{cls.__name__}".\n'
                        f'Custom type-hints might be one reason why "{classvar}" is not recognized. '
                        f'If you have added custom type-hints, please try removing them and set "{classvar}" like so: `{classvar} = <value>`'
                    )
            else:
                setattr(cls, classvar, classvar_values.__getattribute__(classvar))

    @classmethod
    def _pre_registration_hook(cls):
        pass

    @classmethod
    def __register_subclass(cls):
        subclass_name: str = str(cls.__name__).strip()
        cls.__add_to_registry(subclass_name, cls)  ## Always register subclass name
        for k in set(as_list(cls.aliases) + as_list(cls._registry_keys())):
            if k is not None:
                cls.__add_to_registry(k, cls)

    @classmethod
    @validate_arguments
    def __add_to_registry(cls, key: Any, subclass: Type):
        subclass_name: str = subclass.__name__
        if isinstance(key, (str, AutoEnum)):
            ## Case-insensitive matching:
            keys_to_register: List[str] = [String.str_normalize(key)]
        elif isinstance(key, tuple):
            keys_to_register: List[Tuple] = [
                tuple(
                    ## Case-insensitive matching:
                    String.str_normalize(k) if isinstance(k, (str, AutoEnum)) else k
                    for k in key
                )
            ]
        else:
            keys_to_register: List[Any] = [key]
        for k in keys_to_register:
            if k not in cls._registry:
                cls._registry[k] = {subclass_name: subclass}
                continue
            ## Key is in the registry
            registered: Dict[str, Type] = cls._registry[k]
            registered_names: Set[str] = set(registered.keys())
            assert len(registered_names) > 0, f"Invalid state: key {k} is registered to an empty dict"
            if subclass_name in registered_names and cls._allow_subclass_override is False:
                raise KeyError(
                    f"A subclass with name {subclass_name} is already registered against key {k} for registry under "
                    f"{cls._registry_base_class}; overriding subclasses is not permitted."
                )
            elif subclass_name not in registered_names and cls._allow_multiple_subclasses is False:
                assert len(registered_names) == 1, (
                    f"Invalid state: _allow_multiple_subclasses is False but we have multiple subclasses registered "
                    f"against key {k}"
                )
                raise KeyError(
                    f"Key {k} already is already registered to subclass {next(iter(registered_names))}; registering "
                    f"multiple subclasses to the same key is not permitted."
                )
            cls._registry[k] = {
                **registered,
                ## Add or override the subclass names
                subclass_name: subclass,
            }

    @classmethod
    def get_subclass(
        cls,
        key: Any,
        raise_error: bool = True,
        *args,
        **kwargs,
    ) -> Optional[Union[Type, List[Type]]]:
        if isinstance(key, (str, AutoEnum)):
            Subclass: Optional[Dict[str, Type]] = cls._registry.get(String.str_normalize(key))
        else:
            Subclass: Optional[Dict[str, Type]] = cls._registry.get(key)
        if Subclass is None:
            if raise_error:
                raise KeyError(
                    f'Could not find subclass of {cls} using key: "{key}" (type={type(key)}). '
                    f"Available keys are: {set(cls._registry.keys())}"
                )
            return None
        if len(Subclass) == 1:
            return next(iter(Subclass.values()))
        return list(Subclass.values())

    @classmethod
    def subclasses(cls, keep_abstract: bool = False) -> Set[Type]:
        available_subclasses: Set[Type] = set()
        for k, d in cls._registry.items():
            for subclass in d.values():
                if subclass == cls._registry_base_class:
                    continue
                if is_abstract(subclass) and keep_abstract is False:
                    continue
                if isinstance(subclass, type) and issubclass(subclass, cls):
                    available_subclasses.add(subclass)
        return available_subclasses

    @classmethod
    def remove_subclass(cls, subclass: Union[Type, str]):
        name: str = subclass
        if isinstance(subclass, type):
            name: str = subclass.__name__
        for k, d in cls._registry.items():
            for subclass_name, subclass in list(d.items()):
                if String.str_normalize(subclass_name) == String.str_normalize(name):
                    d.pop(subclass_name, None)

    @classmethod
    def _registry_keys(cls) -> Optional[Union[List[Any], Any]]:
        return None


## Ref: https://stackoverflow.com/q/6760685/4900327, Method 2 base class.
## The metaclass method in the above link did not work well with multiple inheritance.
class Singleton:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.__instance, cls):
            cls.__instance = super(Singleton, cls).__new__(cls)
        return cls.__instance

    @classproperty
    def instance(cls):
        return cls.__instance


ParametersSubclass = TypeVar("ParametersSubclass", bound="Parameters")


class Parameters(BaseModel, ABC):
    ## Ref on Pydantic + ABC: https://pydantic-docs.helpmanual.io/usage/models/#abstract-base-classes
    ## Needed to work with Registry.alias...this needs to be on a subclass of `BaseModel`.
    aliases: ClassVar[Tuple[str, ...]] = tuple()
    dict_exclude: ClassVar[Tuple[str, ...]] = tuple()

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except Exception as e:
            raise ValueError(
                f'Cannot create Pydantic instance of type "{self.class_name}".'
                f"\nEncountered exception: {String.format_exception_msg(e)}"
            )

    @classproperty
    def class_name(cls) -> str:
        return str(cls.__name__)  ## Will return the child class name.

    @classmethod
    def param_names(cls, **kwargs) -> Set[str]:
        # superclass_params: Set[str] = set(super(Parameters, cls).schema(**kwargs)['properties'].keys())
        class_params: Set[str] = set(cls.schema(**kwargs)["properties"].keys())
        return class_params  # .union(superclass_params)

    @classmethod
    def param_default_values(cls, **kwargs) -> Dict:
        return {
            param: param_schema["default"]
            for param, param_schema in cls.schema(**kwargs)["properties"].items()
            if "default" in param_schema  ## The default value might be None
        }

    @classmethod
    def _clear_extra_params(cls, params: Dict) -> Dict:
        return {k: v for k, v in params.items() if k in cls.param_names()}

    def dict(self, *args, exclude: Optional[Any] = None, **kwargs) -> Dict:
        exclude: Set[str] = as_set(get_default(exclude, [])).union(as_set(self.dict_exclude))
        return super(Parameters, self).dict(*args, exclude=exclude, **kwargs)

    def json(self, *args, encoder: Optional[Any] = None, indent: Optional[int] = None, **kwargs) -> str:
        if encoder is None:
            encoder = functools.partial(json.dumps, cls=NeverFailJsonEncoder, indent=indent)
        return super(Parameters, self).json(*args, encoder=encoder, **kwargs)

    @classproperty
    def _constructor(cls) -> ParametersSubclass:
        return cls

    def __str__(self) -> str:
        params_str: str = self.json(indent=4)
        out: str = f"{self.class_name} with params:\n{params_str}"
        return out

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

    @staticmethod
    def _convert_params(Class: Type[BaseModel], d: Union[Type[BaseModel], Dict]):
        if type(d) == Class:
            return d
        if isinstance(d, BaseModel):
            return Class(**d.dict(exclude=None))
        if d is None:
            return Class()
        if isinstance(d, dict):
            return Class(**d)
        raise NotImplementedError(f"Cannot convert object of type {type(d)} to {Class.__class__}")

    def update_params(self, **new_params) -> Generic[ParametersSubclass]:
        ## Since Parameters class is immutable, we create a new one:
        overidden_params: Dict = {
            **self.dict(exclude=None),
            **new_params,
        }
        return self._constructor(**overidden_params)

    def copy(self, **kwargs) -> Generic[ParametersSubclass]:
        return super(Parameters, self).copy(**kwargs)

    def clone(self, **kwargs) -> Generic[ParametersSubclass]:
        return self.copy(**kwargs)


class UserEnteredParameters(Parameters):
    """
    Case-insensitive Parameters class.
    Use this for configs classes where you expect to read from user-entered input, which might have any case.
    IMPORTANT: the param names in the subclass must be in LOWERCASE ONLY.
    Ref: https://github.com/samuelcolvin/pydantic/issues/1147#issuecomment-571109376
    """

    @root_validator(pre=True)
    def convert_params_to_lowercase(cls, params: Dict):
        return {str(k).strip().lower(): v for k, v in params.items()}


class MutableParameters(Parameters):
    class Config(Parameters.Config):
        ## Ref on mutability: https://pydantic-docs.helpmanual.io/usage/models/#faux-immutability
        allow_mutation = True


class MutableUserEnteredParameters(UserEnteredParameters, MutableParameters):
    pass


class MappedParameters(Parameters, ABC):
    """
    Allows creation of a Parameters instance by mapping from a dict.
    From this dict, the 'name' key will be used to look up the cls._mapping dictionary, and retrieve the corresponding
    class. This class will be instantiated using the other values in the dict.
    """

    _mapping: ClassVar[Dict[Union[Tuple[str, ...], str], Any]]

    class Config(Parameters.Config):
        extra = Extra.allow

    name: constr(min_length=1)
    args: Tuple = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not isinstance(cls._mapping, dict) or len(cls._mapping) == 0:
            raise ValueError(f"Lookup must be a non-empty dict; found: {cls._mapping}")
        for key, val in list(cls._mapping.items()):
            if is_list_like(key):
                for k in key:
                    cls._mapping[String.str_normalize(k)] = val
            else:
                cls._mapping[String.str_normalize(key)] = val

    @root_validator(pre=True)
    def check_mapped_params(cls, params: Dict) -> Dict:
        if String.str_normalize(params["name"]) not in cls._mapping:
            raise ValueError(
                f'''`name`="{params["name"]}" was not found in the lookup. '''
                f"""Valid values for `name`: {set(cls._mapping.keys())}"""
            )
        return params

    def dict(self, *args, exclude: Optional[Any] = None, **kwargs) -> Dict:
        params: Dict = super(Parameters, self).dict(*args, exclude=exclude, **kwargs)
        if exclude is not None and "name" in exclude:
            params.pop("name", None)
        else:
            params["name"] = self.name
        return params

    def __str__(self) -> str:
        params_str: str = self.json(indent=4)
        out: str = f"{self.class_name} with params:\n{params_str}"
        return out

    @classmethod
    def from_call_str(cls, call_str: str) -> Any:
        args, kwargs = call_str_to_params(call_str)
        return cls(args=args, **kwargs)

    def mapped_callable(self) -> Any:
        return self._mapping[String.str_normalize(self.name)]

    @property
    def kwargs(self) -> Dict:
        return self.dict(exclude={"name", "args"} | set(self.dict_exclude))

    def to_call_str(self) -> str:
        args: List = list(self.args)
        kwargs: Dict = self.kwargs
        callable: Callable = self.mapped_callable()
        if is_function(callable) or isinstance(callable, type):
            callable_name: str = callable.__name__
        else:
            callable_name: str = str(callable)
        return params_to_call_str(
            callable_name=callable_name,
            args=args,
            kwargs=kwargs,
        )

    @classmethod
    @safe_validate_arguments
    def of(
        cls,
        name: Optional[Union[Parameters, Dict, str]],
        **params,
    ) -> Optional[Any]:
        if name is None:
            return None
        if isinstance(name, cls):
            return name
        if isinstance(name, dict):
            return cls(**name)
        if isinstance(name, str):
            if "(" in name or ")" in name:
                return cls.from_call_str(name)
            else:
                return cls(**{"name": name, **params})
        raise ValueError(f"Unsupported value for `name`: {name}")

    def initialize(self, **kwargs) -> Any:
        return self.mapped_callable()(*self.args, **self.kwargs, **kwargs)
