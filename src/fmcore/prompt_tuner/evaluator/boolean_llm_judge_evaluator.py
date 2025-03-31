from typing import Dict
from langchain_core.messages import BaseMessage

from fmcore.prompt_tuner.evaluator.base_evaluator import BaseEvaluator, O
from fmcore.prompt_tuner.evaluator.enums.evaluator_enums import EvaluatorType
from fmcore.prompt_tuner.evaluator.types.evaluator_params_types import BooleanLLMJudgeParams
from fmcore.prompt_tuner.evaluator.types.evaluator_types import (
    EvaluatorConfig,
    BooleanLLMJudgeInput,
    BooleanLLMJudgeOutput,
)
from fmcore.transformers.criteria_checker_transformer import CriteriaChecker
from fmcore.transformers.llm_predictor_transformer import LLMPredictor
from fmcore.transformers.llm_response_json_transformer import LLMResponseJsonTransformer
from fmcore.transformers.text_prompt_transformer import TextPromptTransformer


class BooleanLLMJudgeEvaluator(BaseEvaluator[BooleanLLMJudgeInput, BooleanLLMJudgeOutput]):
    """
    An evaluator that uses an LLM to judge boolean criteria based on a given prompt template and context.

    Note: This is a throw away interface and should be replaced with Transformer Pipeline

    """

    aliases = [EvaluatorType.BOOLEAN_LLM_JUDGE]

    text_prompt_transformer: TextPromptTransformer
    llm_predictor: LLMPredictor
    llm_response_json_extractor: LLMResponseJsonTransformer
    criteria_checker: CriteriaChecker

    @classmethod
    def _get_constructor_parameters(cls, *, evaluator_config: EvaluatorConfig) -> dict:
        """
        Extracts and constructs the parameters required to initialize the evaluator.

        Args:
            evaluator_config (EvaluatorConfig): Configuration object containing evaluator parameters.

        Returns:
            dict: A dictionary containing initialized parameters (`config`, `llm`, `prompt_template`, `criteria`).
        """
        boolean_llm_judge_params: BooleanLLMJudgeParams = evaluator_config.evaluator_params
        text_prompt_transformer: TextPromptTransformer = TextPromptTransformer(
            prompt_template=boolean_llm_judge_params.prompt
        )
        llm_predictor: LLMPredictor = LLMPredictor(llm_config=boolean_llm_judge_params.llm_config)
        llm_response_json_extractor: LLMResponseJsonTransformer = LLMResponseJsonTransformer()
        criteria_checker: CriteriaChecker = CriteriaChecker(criteria=boolean_llm_judge_params.criteria)

        return {
            "config": evaluator_config,
            "text_prompt_transformer": text_prompt_transformer,
            "llm_predictor": llm_predictor,
            "llm_response_json_extractor": llm_response_json_extractor,
            "criteria_checker": criteria_checker,
        }

    def evaluate(self, data: BooleanLLMJudgeInput) -> O:
        """
        Processes the input data by rendering the prompt template with the given context,
        sends the generated prompt to the LLM for evaluation, and determines the result
        based on the response.

        The process follows these steps:
        1. Render the prompt using the input context.
        2. Send the generated prompt to the LLM, expecting a JSON-formatted response.
        3. Parse the response and evaluate whether it meets the predefined criteria.

        Args:
            data (BooleanLLMJudgeInput): Input data containing context for evaluation.

        Returns:
            BooleanLLMJudgeOutput: Evaluation result as a boolean decision.
        """
        messages: BaseMessage = self.text_prompt_transformer.transform(data=data.context)
        llm_response: BaseMessage = self.llm_predictor.transform(data=[messages])
        json_response: Dict = self.llm_response_json_extractor.transform(data=llm_response.content)
        decision = self.criteria_checker.transform(data=json_response)

        return BooleanLLMJudgeOutput(decision=decision)

    async def aevaluate(self, data: BooleanLLMJudgeInput) -> O:
        """
        Asynchronous version of `evaluate` that processes the input data and assesses the response.

        Args:
            data (BooleanLLMJudgeInput): Input data containing context for evaluation.

        Returns:
            BooleanLLMJudgeOutput: Evaluation result as a boolean decision.
        """
        messages: BaseMessage = await self.text_prompt_transformer.atransform(data=data.context)
        llm_response: BaseMessage = await self.llm_predictor.atransform(data=[messages])
        json_response: Dict = await self.llm_response_json_extractor.atransform(data=llm_response.content)
        decision = await self.criteria_checker.atransform(data=json_response)

        return BooleanLLMJudgeOutput(decision=decision)
