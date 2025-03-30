from typing import Dict, List

from asteval import Interpreter
from jinja2 import Template
from json_repair import json_repair
from langchain_core.messages import BaseMessage, HumanMessage

from fmcore.evaluator.base_evaluator import BaseEvaluator, I, O
from fmcore.evaluator.enums.evaluator_enums import EvaluatorType
from fmcore.evaluator.types.evaluator_params_types import BooleanLLMJudgeParams
from fmcore.evaluator.types.evaluator_types import (
    EvaluatorConfig,
    BooleanLLMJudgeInput,
    BooleanLLMJudgeOutput,
)
from fmcore.llm import BaseLLM


class BooleanLLMJudgeEvaluator(BaseEvaluator[BooleanLLMJudgeInput, BooleanLLMJudgeOutput]):
    """
    An evaluator that uses an LLM to judge boolean criteria based on a given prompt template and context.
    """

    aliases = [EvaluatorType.BOOLEAN_LLM_JUDGE]

    llm: BaseLLM
    prompt_template: Template
    criteria: str

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
        llm = BaseLLM.of(llm_config=boolean_llm_judge_params.llm_config)

        # Parsing a Jinja template is an expensive operation. We benchmarked two approaches:
        # 1. Creating a new template for each render before rendering (100k renders took 284.8903 sec).
        # 2. Using a single pre-compiled template and rendering it multiple times (100k renders took 3.4328 sec).
        # The second approach resulted in an 85x performance improvement.

        # Since the template remains unchanged across multiple evaluators, we create it once
        # and reuse it throughout the evaluator to optimize performance.
        prompt_template = Template(source=boolean_llm_judge_params.prompt, autoescape=True)
        criteria = boolean_llm_judge_params.criteria

        return {
            "config": evaluator_config,
            "llm": llm,
            "prompt_template": prompt_template,
            "criteria": criteria,
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
        prompt = self.prompt_template.render(data.context)
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages=messages)
        return self.evaluate_criteria(response=response)

    async def aevaluate(self, data: BooleanLLMJudgeInput) -> O:
        """
        Asynchronous version of `evaluate` that processes the input data and assesses the response.

        Args:
            data (BooleanLLMJudgeInput): Input data containing context for evaluation.

        Returns:
            BooleanLLMJudgeOutput: Evaluation result as a boolean decision.
        """
        prompt = self.prompt_template.render(data.context)
        messages = [BaseMessage(content=prompt)]
        response = await self.llm.ainvoke(messages=messages)
        return self.evaluate_criteria(response=response)

    def evaluate_criteria(self, response: BaseMessage) -> BooleanLLMJudgeOutput:
        """
        Parses the LLM's response, extracts the JSON data, and evaluates it against the specified criteria.

        The evaluation follows these steps:
        1. Parse the JSON string returned by the LLM using `json_repair` to handle any malformed JSON.
        2. Use `asteval` to create an interpreter that evaluates the user's criteria.
        3. Inject the parsed JSON data into the interpreter’s symbol table.
        4. Evaluate the criteria as an expression against the extracted values.
        5. Return the decision as a boolean result.

        Args:
            response (BaseMessage): Response message from the LLM, expected to be a JSON-formatted string.

        Returns:
            BooleanLLMJudgeOutput: Evaluation result containing the boolean decision.
        """

        # Parse the LLM's response content into a dictionary, repairing any malformed JSON.
        # We use the out-of-the-box 'json_repair' library instead of implementing our own parser,
        # as it efficiently handles common JSON formatting issues (e.g., missing quotes, misplaced commas).
        # https://pypi.org/project/json-repair/
        # We might be changing this in the future if we find any robust impementations
        evaluation_response: Dict = json_repair.loads(response.content)

        # AST interpreters are not inherently thread-safe, as they maintain an internal symbol table
        # that is modified during execution. To ensure correctness, we instantiate a new Interpreter
        # for each evaluation instead of sharing a global instance.

        # Using a shared Interpreter would require synchronization mechanisms such as locks or
        # thread-local storage to prevent concurrent modifications to the symbol table. However,
        # benchmarking showed that even with optimizations, a shared, thread-safe implementation
        # was at best only **30% faster** than creating a new instance per evaluation.

        # Given that Interpreter instantiation is lightweight and avoids race conditions, the optimal
        # approach is to create a new instance for each evaluation, populate its symbol table with
        # the extracted values, and execute the criteria expression while maintaining correctness and performance.

        expression_evaluator = Interpreter()
        expression_evaluator.symtable.update(evaluation_response)
        decision: bool = expression_evaluator(self.criteria)
        return BooleanLLMJudgeOutput(decision=decision)
