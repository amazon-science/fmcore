from typing import Optional, Union

from fmcore.llm.types.llm_types import LLMConfig, DistributedLLMConfig
from fmcore.types.mixins_types import Mixin
from fmcore.types.typed import MutableTyped


class LLMConfigMixin(MutableTyped, Mixin):
    """
    Mixin for LLM configuration.

    Attributes:
        llm_config (Optional[LLMConfig]): The LLM configuration object.
    """

    llm_config: Union[LLMConfig, DistributedLLMConfig]
