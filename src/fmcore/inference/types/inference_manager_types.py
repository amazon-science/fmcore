from abc import ABC
from typing import Dict, Union

from bears.util import Registry
from pydantic import model_validator

from fmcore.llm.types.llm_types import DistributedLLMConfig, LLMConfig
from fmcore.types.typed import MutableTyped


class InferenceManagerParams(MutableTyped, Registry, ABC):
    """
    Abstract base class for defining parameters specific to an inference manager.

    Subclasses of this class should define manager-specific parameters,
    and can be dynamically resolved using the `inference_manager_type` key.
    """

    pass


class InferenceManagerConfig(MutableTyped):
    """
    Configuration class for initializing an inference manager.

    This class holds essential information like the type of inference manager to use,
    the LLM configuration, and the associated inference manager parameters.

    Attributes:
        inference_manager_type (str): Key identifying which inference manager subclass to use.
        llm_config (Union[LLMConfig, DistributedLLMConfig]): Configuration for the LLM(s).
        inference_manager_params (InferenceManagerParams): Manager-specific parameters.
    """

    inference_manager_type: str
    llm_config: Union[LLMConfig, DistributedLLMConfig]
    inference_manager_params: InferenceManagerParams

    @model_validator(mode="before")
    def validate(cls, values: Dict):
        """
        Validates and converts `inference_manager_params` from a raw dict to the correct subclass instance.

        Args:
            values (Dict): Raw dictionary of values passed to the model.

        Returns:
            Dict: Updated values with the correct `InferenceManagerParams` subclass instance.
        """
        inference_manager_type = values.get("inference_manager_type")
        inference_manager_params = values.get("inference_manager_params")

        if isinstance(inference_manager_params, Dict):
            InferenceManagerParamsClass = InferenceManagerParams.get_subclass(key=inference_manager_type)
            values["inference_manager_params"] = InferenceManagerParamsClass(**inference_manager_params)

        return values
