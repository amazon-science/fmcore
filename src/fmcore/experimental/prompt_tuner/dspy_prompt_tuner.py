import dspy
from pandas import DataFrame
from typing import Callable, Dict, Optional, List, Tuple, Type

from dspy.teleprompt import Teleprompter
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy import Signature, Module


from fmcore.experimental.metrics.base_metric import BaseMetric
from fmcore.experimental.prompt_tuner.base_prompt_tuner import BasePromptTuner
from fmcore.experimental.types.enums.prompt_tuner_enums import PromptTunerFramework
from fmcore.experimental.types.prompt_tuner_types import PromptTunerConfig

from fmcore.experimental.adapters.dspy_adapter import DSPyLLMAdapter
from fmcore.experimental.utils.introspection_utils import IntrospectionUtils
from fmcore.experimental.prompt_tuner.utils.dspy_utils import DSPyUtils
from py_expression_eval import Parser
from asteval import Interpreter




class DSPyPromptTuner(BasePromptTuner):
    aliases = [PromptTunerFramework.DSPY]
    student: dspy.LM
    teacher: Optional[dspy.LM]
    optimizer_metric: BaseMetric

    @classmethod
    def _get_constructor_parameters(cls, *, config: PromptTunerConfig) -> Dict:
        student_model = DSPyLLMAdapter(llm_config=config.optimzer_config.student_config)
        dspy.configure(lm=student_model)

        if config.optimzer_config.teacher_config:
            teacher_model = DSPyLLMAdapter(llm_config=config.optimzer_config.teacher_config)
        else:
            teacher_model = student_model
        optimizer_metric = BaseMetric.of(metric_config=config.optimzer_config.metric_config)

        return {
            "student": student_model,
            "teacher": teacher_model,
            "optimizer_metric": optimizer_metric,
            "config": config,
        }

    def _create_evaluation_function(self):
        """
        Creates an evaluation function that uses the configured metric.

        Returns:
            Evaluation function that takes an example and prediction
        """
        
        # Store criteria once to avoid re-fetching it in each evaluation call
        criteria = self.optimizer_metric.config.metric_params["criteria"]

        def evaluate_func(example: dspy.Example, prediction: dspy.Prediction, trace=None):
            # Get evaluation results
            evaluation_response: dict = DSPyUtils.evaluate(
                example=example, 
                prediction=prediction, 
                metric=self.optimizer_metric
            )

            # Create a new Interpreter instance for each call to ensure thread safety
            expression_evaluator = Interpreter()
            expression_evaluator.symtable.update(evaluation_response)
            return expression_evaluator(criteria)

        return evaluate_func



    def tune(self, data: DataFrame) -> str:
        """
        Tunes a prompt using the configured DSPy optimizer.

        Args:
            data: DataFrame containing the training data
            prompt_config: Configuration containing input and output fields

        Returns:
            The optimized prompt as a string
        """

        # Convert data to DSPy examples
        dspy_examples = DSPyUtils.convert_to_dspy_examples(
            data=data, prompt_config=self.config.prompt_config
        )

        # Create signature and module separately
        signature: Signature = DSPyUtils.create_dspy_signature(
            prompt_config=self.config.prompt_config
        )
        module: Module = DSPyUtils.create_dspy_module(signature=signature)

        # Create evaluation function
        evaluate_func = self._create_evaluation_function()

        # Create optimizer
        optimizer: Teleprompter = DSPyUtils.get_optimizer(
            student=self.student,
            teacher=self.teacher,
            optimzer_config=self.config.optimzer_config,
            evaluate_func=evaluate_func,
        )

        filtered_optimizer_params = IntrospectionUtils.filter_params(
            func=optimizer.compile, params=self.config.optimzer_config.params
        )
        # Compile the module with the optimizer
        optimized_module = optimizer.compile(
            student=module,
            trainset=dspy_examples,
            requires_permission_to_run=False,
            **filtered_optimizer_params,
        )
