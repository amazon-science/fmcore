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
from fmcore.utils.logging_utils import Log


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

    def _process_judge_json_responses(
        self,
        judge_responses: list,
        context: Dict,
        formatted_prompt: BaseMessage,
        llm_response: BaseMessage,
    ) -> bool:
        """
        Process a list of JSON responses from judge responses and return True if any response is True.
        Logs warnings for non-boolean responses and raises an exception if no boolean values are found.

        Args:
            judge_responses (list): List of JSON responses from the judge
            context (Dict): Original input context
            formatted_prompt (BaseMessage): The formatted prompt sent to the LLM
            llm_response (BaseMessage): The raw response from the LLM

        Returns:
            bool: True if any response is True, False if all responses are False

        Raises:
            ValueError: If no boolean values are found in any response
        """
        found_boolean = False
        for judge_response in judge_responses:
            decision = self.criteria_checker.map(judge_response)
            if isinstance(decision, bool):
                found_boolean = True
                if decision:
                    return True
            else:
                Log.warning(
                    "[NON-BOOLEAN RESPONSE]\t\t ->"
                    f"[INPUT DATA]: {context}\t\t ->"
                    f"[PROMPT]: {self.config.evaluator_params.prompt}\t\t ->"
                    f"[FORMATTED MESSAGE]: {formatted_prompt}\t\t ->"
                    f"[LLM RESPONSE]: {llm_response}\t\t ->"
                    f"[JUDGE RESPONSE]: {judge_response}\t\t ->"
                    f"[DECISION]: {decision}"
                )

        if not found_boolean:
            raise ValueError("None of the decisions are boolean values")

        return False

    def evaluate(self, data: Dict) -> bool:
        """
        Processes the input data using the llm_as_a_judge_boolean_mapper to evaluate the context.
        Checks criteria on each element of the response list and returns True if any element evaluates to True.
        Throws an exception if none of the responses are boolean values.
        Logs a warning for each non-boolean response.

        Args:
            data (BooleanLLMJudgeInput): Input data containing context for evaluation.

        Returns:
            bool: True if any element in the response list evaluates to True, False otherwise.

        Raises:
            ValueError: If none of the responses are boolean values.
        """
        formatted_prompt = llm_response = judge_responses = None

        try:
            formatted_prompt = self.text_prompt_mapper.map(data)
            llm_response = self.llm_inference_mapper.map([formatted_prompt])
            judge_responses = self.json_mapper.map(llm_response.content)

            return self._process_judge_json_responses(
                judge_responses=judge_responses,
                context=data,
                formatted_prompt=formatted_prompt,
                llm_response=llm_response,
            )

        except Exception as e:
            Log.error(
                "[SYNC EVALUATION ERROR]\t\t ->"
                f"[INPUT DATA]: {data}\t\t ->"
                f"[PROMPT]: {self.config.evaluator_params.prompt}\t\t ->"
                f"[FORMATTED MESSAGE]: {formatted_prompt}\t\t ->"
                f"[LLM RESPONSE]: {llm_response}\t\t ->"
                f"[JUDGE RESPONSES]: {judge_responses}\t\t ->"
                f"[ERROR]: {e}"
            )
            raise

    async def aevaluate(self, data: Dict) -> bool:
        """
        Asynchronous version of `evaluate` that processes the input data.
        Checks criteria on each element of the response list and returns True if any element evaluates to True.
        Throws an exception if none of the responses are boolean values.
        Logs a warning for each non-boolean response.

        Args:
            data (BooleanLLMJudgeInput): Input data containing context for evaluation.

        Returns:
            bool: True if any element in the response list evaluates to True, False otherwise.

        Raises:
            ValueError: If none of the responses are boolean values.
        """
        formatted_prompt = llm_response = judge_responses = None

        try:
            formatted_prompt = await self.text_prompt_mapper.amap(data)
            llm_response = await self.llm_inference_mapper.amap([formatted_prompt])
            judge_responses = await self.json_mapper.amap(llm_response.content)

            return self._process_judge_json_responses(
                judge_responses=judge_responses,
                context=data,
                formatted_prompt=formatted_prompt,
                llm_response=llm_response,
            )

        except Exception as e:
            Log.error(
                "[ASYNC EVALUATION ERROR]\t\t->"
                f"[INPUT DATA]: {data}\t\t ->"
                f"[PROMPT]: {self.config.evaluator_params.prompt}\t\t ->"
                f"[FORMATTED MESSAGE]: {formatted_prompt}\t\t ->"
                f"[LLM RESPONSE]: {llm_response}\t\t ->"
                f"[JUDGE RESPONSES]: {judge_responses}\t\t ->"
                f"[ERROR]: {e}"
            )
            raise
