from typing import Dict

from pydantic import model_validator

from fmcore.prompt_tuner.evaluator.enums.evaluator_enums import EvaluatorType
from fmcore.prompt_tuner.evaluator.types.evaluator_params_types import BaseEvaluatorParams
from fmcore.types.typed import MutableTyped


class EvaluatorConfig(MutableTyped):
    """
    Configuration class for different evaluators.

    Attributes:
        evaluator_type (EvaluatorType): The type of evaluator to be used.
        evaluator_params (BaseEvaluatorParams): The parameters required by the evaluator.
    """

    evaluator_type: EvaluatorType
    evaluator_params: BaseEvaluatorParams

    @model_validator(mode="before")
    def parse_provider_params(cls, values: Dict):
        """
        Transforms evaluator_params based on evaluator_type before object creation.

        This method allows clients to register their evaluators dynamically at runtime.
        Each evaluator can have its own run configuration. Pydantic's native discriminator
        is avoided as it requires a union of all possible classes, which is not feasible
        at runtime when users can extend interfaces to define custom evaluators.

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


class LLMAsAJudgeInput(MutableTyped):
    """
    Input data structure for LLM as a Judge.

    Attributes:
        context (Dict): The context information required for evaluation.
    """

    context: Dict


class LLMAsAJudgeBooleanOutput(MutableTyped):
    """
    Output data structure for Boolean outputs from LLM as a Judge.

    Attributes:
        decision (bool): The result of the evaluation, indicating whether the response
        meets the specified criteria.
    """

    decision: bool
