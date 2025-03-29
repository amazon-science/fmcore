from abc import ABC
from typing import TypeVar

from pydantic import BaseModel, Extra

TypedSubclass = TypeVar("TypedSubclass", bound="Typed")


class Typed(BaseModel, ABC):
    """
    Ref on Pydantic + ABC: https://pydantic-docs.helpmanual.io/usage/models/#abstract-base-classes
    To serialize any object subclassing this in camel case convention, use:
    """

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


class MutableTyped(Typed, ABC):
    ## Ref on Pydantic + ABC: https://pydantic-docs.helpmanual.io/usage/models/#abstract-base-classes

    class Config(Typed.Config):
        ## Ref for Pydantic mutability: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.frozen
        frozen = False
        ## Ref of validating assignment: https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.validate_assignment
        validate_assignment = True

        # The UI currently sends a lot of additional fields which are not required
        # Allowing those fields for now to avoid RTEs
        extra = Extra.allow
