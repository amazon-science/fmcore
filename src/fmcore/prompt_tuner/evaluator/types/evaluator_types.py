from typing import Dict

from pydantic import model_validator, SerializeAsAny

from fmcore.prompt_tuner.evaluator.types.evaluator_params_types import BaseEvaluatorParams
from fmcore.types.typed import MutableTyped


class EvaluatorConfig(MutableTyped):
    """
    Configuration class for different evaluators.

    Attributes:
        evaluator_type (EvaluatorType): The type of evaluator to be used.
        evaluator_params (BaseEvaluatorParams): The parameters required by the evaluator.

    Note:
        Follows the same principles as `LLMConfig`.
    """

    evaluator_type: str
    evaluator_params: SerializeAsAny[BaseEvaluatorParams]

    @model_validator(mode="before")
    def parse_provider_params(cls, values: Dict):
        """
        Transforms evaluator_params based on evaluator_type before object creation.

        This method allows clients to register their evaluators dynamically at runtime.
        Each evaluator can have its own run configuration.

        Args:
            values (Dict): The input dictionary containing evaluator_type and evaluator_params.

        Returns:
            Dict: The transformed values with evaluator_params converted to the appropriate class.
        """
        if isinstance(values.get("evaluator_params"), Dict):  # Only transform if it's a dict
            values["evaluator_params"] = BaseEvaluatorParams.from_dict(
                evaluator_type=values.get("evaluator_type"), evaluator_params=values.get("evaluator_params")
            )
        return values
