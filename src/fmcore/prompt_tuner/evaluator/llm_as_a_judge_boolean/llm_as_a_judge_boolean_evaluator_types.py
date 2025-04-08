from fmcore.llm.mixins.llm_mixins import LLMConfigMixin
from fmcore.prompt_tuner.evaluator.types.evaluator_params_types import BaseEvaluatorParams


class BooleanLLMJudgeParams(BaseEvaluatorParams, LLMConfigMixin):
    """
    Parameters for the Boolean LLM Judge evaluator.

    This evaluator takes a prompt, criteria, and an LLM instance to validate whether
    the LLM's response adheres to the given criteria.

    Attributes:
        prompt (str): The prompt used for the LLM-based evaluation.
        criteria (str): The criteria against which the evaluation is performed.
    """

    aliases = ["LLM_AS_A_JUDGE_BOOLEAN"]

    prompt: str
    criteria: str
