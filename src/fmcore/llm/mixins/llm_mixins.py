from typing import Optional

from fmcore.llm.types.llm_types import LLMConfig
from fmcore.types.mixins_types import Mixin
from fmcore.types.typed import MutableTyped


class LLMConfigMixin(MutableTyped, Mixin):
    """
    Mixin for LLM configuration.

    Attributes:
        llm_config (Optional[LLMConfig]): The LLM configuration object.
    """

    llm_config: LLMConfig
