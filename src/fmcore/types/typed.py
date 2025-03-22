from typing import (
    Set,
    Dict,
    Generic,
    TypeVar,
)
from abc import ABC
from pydantic import BaseModel
from pydantic.alias_generators import to_camel

TypedSubclass = TypeVar("TypedSubclass", bound="Typed")


class Typed(BaseModel, ABC):
    """
    Ref on Pydantic + ABC: https://pydantic-docs.helpmanual.io/usage/models/#abstract-base-classes
    To serialize any object subclassing this in camel case convention, use:

    >>> json.dumps(pydantic_object.model_dump(mode="json", by_alias=True)

    model_dump would return snake cased (or as defined in class) attribute names without 'by_alias' arg.
    """

    # TODO on aashok@: Check pydantic for snake case method
    class Config:
        ## Ref for Pydantic mutability: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.frozen
        frozen = True
        ## Ref for Extra.forbid: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.extra
        extra = "forbid"

        ## Ref for Pydantic private attributes: https://pydantic-docs.helpmanual.io/usage/models/#private-model-attributes
        ## Note: in Pydantic 2, underscore_attrs_are_private is true by default: https://docs.pydantic.dev/1.10/blog/pydantic-v2-alpha/#changes-to-config
        ## underscore_attrs_are_private = True

        ## Validates default values. Ref: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.validate_default
        validate_default = True
        ## Validates return values. Ref: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.validate_return
        validate_return = True

        ## Validates typing via `isinstance` check. Ref: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.arbitrary_types_allowed
        arbitrary_types_allowed = True

        ## to take care of JSON serialization and deserialization for converting to camel case (the default JSON convention used in CSA).
        ## Refer to class level comment for how to serialize
        alias_generator = to_camel

        ## Adding this so that constructor calls with snake case variable names work as well
        populate_by_name = True

    def class_name(self) -> str:
        return str(self.__class__.__name__)  ## Will return the child class name.

    @classmethod
    def param_names(cls, **kwargs) -> Set[str]:
        # superclass_params: Set[str] = set(super(Typed, cls).schema(**kwargs)['properties'].keys())
        class_params: Set[str] = set(cls.schema(**kwargs)["properties"].keys())
        return class_params

    @classmethod
    def param_default_values(cls, **kwargs) -> Dict:
        return {
            param: param_schema["default"]
            for param, param_schema in cls.schema(**kwargs)["properties"].items()
            if "default" in param_schema  ## The default value might be None
        }

    def __str__(self) -> str:
        params_str: str = self.model_dump_json(indent=4)
        out: str = f"{self.class_name()} with params:\n{params_str}"
        return out

    def copy(self, **kwargs) -> Generic[TypedSubclass]:  # type: ignore
        # TODO (@adivekar): Remove the below type: ignore comment which ignores mypy
        return super(Typed, self).copy(**kwargs)


class MutableTyped(Typed, ABC):
    ## Ref on Pydantic + ABC: https://pydantic-docs.helpmanual.io/usage/models/#abstract-base-classes

    class Config(Typed.Config):
        ## Ref for Pydantic mutability: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.frozen
        frozen = False
        ## Ref of validating assignment: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.validate_assignment
        validate_assignment = True
