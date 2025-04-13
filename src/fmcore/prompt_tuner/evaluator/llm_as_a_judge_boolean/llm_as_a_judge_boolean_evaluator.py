from typing import Dict
from langchain_core.messages import BaseMessage

from fmcore.prompt_tuner.evaluator.llm_as_a_judge_boolean.llm_as_a_judge_boolean_evaluator_types import (
    BooleanLLMJudgeParams,
)
from fmcore.prompt_tuner.evaluator.base_evaluator import BaseEvaluator
from fmcore.prompt_tuner.evaluator.types.evaluator_types import EvaluatorConfig
from fmcore.llm.base_llm import BaseLLM
from fmcore.mapper.text_prompt_mapper import TextPromptMapper
from fmcore.mapper.llm_response_json_mapper import LLMResponseJsonMapper
from fmcore.mapper.criteria_checker_mapper import CriteriaCheckerMapper
from fmcore.mapper.llm_inference_mapper import LLMInferenceMapper


class LLMAsJudgeBooleanEvaluator(BaseEvaluator[Dict, bool]):
    """
    An evaluator that processes input data in the form of a dictionary (Dict) and returns
    a boolean (bool) decision based on a judgment criterion evaluated by a large language model (LLM).

    This evaluator is designed to assess a given context or criteria encoded within the input dictionary
    and produce a binary decision (True or False). The core functionality involves:

    1. Mapping the input dictionary to a prompt template using `text_prompt_mapper`.
    2. Feeding the formatted prompt into an LLM using `llm_inference_mapper` for evaluation.
    3. Parsing the LLM's response into structured JSON via `json_mapper`.
    4. Applying `criteria_checker` to the parsed JSON to make a final boolean judgment.

    The transformation of input data from a raw dictionary to a boolean output makes this evaluator
    particularly suited for use cases such as rule-based decision making, automated validation, or
    context-dependent boolean classification tasks.
    """

    aliases = ["LLM_AS_A_JUDGE_BOOLEAN"]

    text_prompt_mapper: TextPromptMapper
    llm_inference_mapper: LLMInferenceMapper
    json_mapper: LLMResponseJsonMapper
    criteria_checker: CriteriaCheckerMapper

    @classmethod
    def _get_instance(cls, *, evaluator_config: EvaluatorConfig) -> "LLMAsJudgeBooleanEvaluator":
        """
        Factory method to create an instance of LLMAsJudgeBooleanEvaluator using the provided configuration.

        This method extracts evaluator-specific parameters, initializes all required components
        (such as mappers and the LLM), and returns a fully constructed evaluator instance.

        Args:
            evaluator_config (EvaluatorConfig): The configuration object containing evaluator parameters.

        Returns:
            LLMAsJudgeBooleanEvaluator: A fully initialized evaluator instance.
        """

        boolean_llm_judge_params: BooleanLLMJudgeParams = evaluator_config.evaluator_params
        # Create required mappers
        text_prompt_mapper = TextPromptMapper(prompt_template=boolean_llm_judge_params.prompt)
        llm_inference_mapper = LLMInferenceMapper(
            llm=BaseLLM.of(llm_config=boolean_llm_judge_params.llm_config)
        )
        json_mapper = LLMResponseJsonMapper()
        criteria_checker = CriteriaCheckerMapper(criteria=boolean_llm_judge_params.criteria)

        return LLMAsJudgeBooleanEvaluator(
            config=evaluator_config,
            text_prompt_mapper=text_prompt_mapper,
            llm_inference_mapper=llm_inference_mapper,
            json_mapper=json_mapper,
            criteria_checker=criteria_checker,
        )

    def evaluate(self, data: Dict) -> bool:
        """
        Processes the input data by using the llm_as_a_judge_boolean_mapper to evaluate the context.

        Args:
            data (BooleanLLMJudgeInput): Input data containing context for evaluation.

        Returns:
            BooleanLLMJudgeOutput: Evaluation result as a boolean decision.
        """
        # Format the context into messages using the template
        formatted_message: BaseMessage = self.text_prompt_mapper.map(data)
        llm_response: BaseMessage = self.llm_inference_mapper.map([formatted_message])
        json_response: Dict = self.json_mapper.map(llm_response.content)
        decision: bool = self.criteria_checker.map(json_response)
        return decision

    async def aevaluate(self, data: Dict) -> bool:
        """
        Asynchronous version of `evaluate` that processes the input data.

        Args:
            data (BooleanLLMJudgeInput): Input data containing context for evaluation.

        Returns:
            BooleanLLMJudgeOutput: Evaluation result as a boolean decision.
        """
        # Format the context into messages using the template
        formatted_message: BaseMessage = await self.text_prompt_mapper.amap(data.context)
        llm_response: BaseMessage = await self.llm_inference_mapper.amap([formatted_message])
        json_response: Dict = await self.json_mapper.amap(llm_response.content)
        decision: bool = await self.criteria_checker.amap(json_response)
        return decision
