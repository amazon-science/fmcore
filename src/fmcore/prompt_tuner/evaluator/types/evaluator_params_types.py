from abc import ABC
from typing import Dict

from bears.util import Registry

from fmcore.prompt_tuner.evaluator.enums.evaluator_enums import EvaluatorType
from fmcore.llm.mixins.llm_mixins import LLMConfigMixin
from fmcore.types.typed import MutableTyped


class BaseEvaluatorParams(MutableTyped, Registry, ABC):
    """
    Base class for evaluator parameters in Evaluators.

    Extends Registry to allow dynamic registration of custom evaluation parameters
    required for custom evaluators.
    """

    @classmethod
    def from_dict(cls, evaluator_type: EvaluatorType, evaluator_params: Dict) -> "BaseEvaluatorParams":
        """
        Creates an instance of the appropriate evaluator parameter subclass from a dictionary.

        This method is required by Pydantic validators to dynamically select parameter classes
        that are registered at runtime. This approach avoids the compile-time limitations of
        Pydantic discriminators.

        Args:
            evaluator_type (EvaluatorType): The type of evaluator to determine the correct subclass.
            evaluator_params (Dict): A dictionary of parameters for the evaluator.

        Returns:
            BaseEvaluatorParams: An instance of the resolved evaluator parameter subclass.
        """
        BaseEvaluatorParamsClass = BaseEvaluatorParams.get_subclass(key=evaluator_type)
        return BaseEvaluatorParamsClass(**evaluator_params)


class BooleanLLMJudgeParams(BaseEvaluatorParams, LLMConfigMixin):
    """
    Parameters for the Boolean LLM Judge evaluator.

    This evaluator takes a prompt, criteria, and an LLM instance to validate whether
    the LLM's response adheres to the given criteria.

    Attributes:
        prompt (str): The prompt used for the LLM-based evaluation.
        criteria (str): The criteria against which the evaluation is performed.
    """

    aliases = [EvaluatorType.LLM_AS_A_JUDGE_BOOLEAN]

    prompt: str
    criteria: str


class ClassificationParams(BaseEvaluatorParams):
    """
    Parameters for the Classification evaluator.

    This evaluator takes a prompt and an LLM instance to classify the LLM's response.

    Attributes:
        prompt (str): The prompt used for the LLM-based evaluation.
    """

    aliases = [EvaluatorType.CLASSIFICATION]

    pass
