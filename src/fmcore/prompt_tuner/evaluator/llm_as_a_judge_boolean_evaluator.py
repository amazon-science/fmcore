from jinja2 import Template
from langchain_core.messages import BaseMessage

from fmcore.prompt_tuner.evaluator.base_evaluator import BaseEvaluator, O
from fmcore.prompt_tuner.evaluator.enums.evaluator_enums import EvaluatorType
from fmcore.prompt_tuner.evaluator.types.evaluator_params_types import BooleanLLMJudgeParams
from fmcore.prompt_tuner.evaluator.types.evaluator_types import (
    EvaluatorConfig,
    BooleanLLMJudgeInput,
    BooleanLLMJudgeOutput,
)
from fmcore.predictor.llm_as_a_judge_boolean_predictor import LLMAsAJudgeBooleanPredictor
from fmcore.llm.base_llm import BaseLLM
from fmcore.mapper.text_prompt_mapper import TextPromptMapper
from fmcore.mapper.llm_response_json_mapper import LLMResponseJsonMapper
from fmcore.mapper.criteria_checker_mapper import CriteriaCheckerMapper


class BooleanLLMJudgeEvaluator(BaseEvaluator[BooleanLLMJudgeInput, BooleanLLMJudgeOutput]):
    """
    An evaluator that uses an LLM to judge boolean criteria based on a given prompt template and context.
    Uses LLMAsAJudgeBooleanPredictor for the core functionality.
    """

    aliases = [EvaluatorType.BOOLEAN_LLM_JUDGE]

    predictor: LLMAsAJudgeBooleanPredictor
    text_prompt_mapper: TextPromptMapper

    @classmethod
    def _get_constructor_parameters(cls, *, evaluator_config: EvaluatorConfig) -> dict:
        """
        Extracts and constructs the parameters required to initialize the evaluator.

        Args:
            evaluator_config (EvaluatorConfig): Configuration object containing evaluator parameters.

        Returns:
            dict: A dictionary containing initialized parameters (`config`, `predictor`, `text_prompt_mapper`).
        """
        boolean_llm_judge_params: BooleanLLMJudgeParams = evaluator_config.evaluator_params
        llm: BaseLLM = BaseLLM.of(llm_config=boolean_llm_judge_params.llm_config)
        # Create required mappers
        text_prompt_mapper = TextPromptMapper(template=Template(boolean_llm_judge_params.prompt))
        json_mapper = LLMResponseJsonMapper()
        criteria_checker = CriteriaCheckerMapper(criteria=boolean_llm_judge_params.criteria)
        
        predictor = LLMAsAJudgeBooleanPredictor(
            llm=llm,
            json_mapper=json_mapper,
            criteria_checker=criteria_checker
        )

        return {
            "config": evaluator_config,
            "predictor": predictor,
            "text_prompt_mapper": text_prompt_mapper,
        }

    def evaluate(self, data: BooleanLLMJudgeInput) -> O:
        """
        Processes the input data by using the LLMAsAJudgeBooleanPredictor to evaluate the context.

        Args:
            data (BooleanLLMJudgeInput): Input data containing context for evaluation.

        Returns:
            BooleanLLMJudgeOutput: Evaluation result as a boolean decision.
        """
        # Format the context into messages using the template
        formatted_message = self.text_prompt_mapper.map(data.context)
        decision = self.predictor.predict([formatted_message])
        return BooleanLLMJudgeOutput(decision=decision)

    async def aevaluate(self, data: BooleanLLMJudgeInput) -> O:
        """
        Asynchronous version of `evaluate` that processes the input data.

        Args:
            data (BooleanLLMJudgeInput): Input data containing context for evaluation.

        Returns:
            BooleanLLMJudgeOutput: Evaluation result as a boolean decision.
        """
        # Format the context into messages using the template
        formatted_message = await self.text_prompt_mapper.amap({"messages": [BaseMessage(content=str(data.context))]})
        decision = await self.predictor.apredict([formatted_message])
        return BooleanLLMJudgeOutput(decision=decision)
