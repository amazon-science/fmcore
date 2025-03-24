import dspy
import pandas as pd
from typing import Any, Callable, Dict, Optional, List, Type

from dspy import Signature, Module
from dspy.evaluate import Evaluate

from fmcore.metrics.base_metric import BaseMetric
from fmcore.prompt_tuner.base_prompt_tuner import BasePromptTuner
from fmcore.prompt_tuner.dspy.datasets.base_dataset import DspyDataset
from fmcore.prompt_tuner.dspy.optimizers.base_dspy_optimizer import BaseDspyOptimizer
from fmcore.prompt_tuner.dspy.utils.commons import DSPyUtils
from fmcore.types.enums.dataset_enums import DatasetType
from fmcore.types.enums.prompt_tuner_enums import PromptTunerFramework
from fmcore.types.prompt_tuner_types import (
    PromptTunerConfig,
    PromptTunerResult,
    PromptEvaluationResult,
    TunedPrompt,
)
from fmcore.prompt_tuner.dspy.adapters.chat_adapter import custom_prepare_instructions
from fmcore.adapters.dspy_adapter import DSPyLLMAdapter

import dspy.adapters.chat_adapter as chat_adapter_module

chat_adapter_module.prepare_instructions = custom_prepare_instructions


class DSPyPromptTuner(BasePromptTuner):
    """
    A prompt tuner implementation using the DSPy framework.

    This class provides functionality to optimize prompts using various DSPy optimizers
    such as MIPROv2 or BootstrapFewShot. It uses a student model for generating responses
    and evaluates them using a configured metric to iteratively improve the prompt.

    Attributes:
        aliases (List[PromptTunerFramework]): Framework identifiers for this tuner.
        student (dspy.LM): The student language model used for prompt optimization.
        teacher (Optional[dspy.LM]): The teacher language model used in some optimization techniques.
        optimizer_metric (BaseMetric): The metric used to evaluate prompt performance.
    """

    aliases = [PromptTunerFramework.DSPY]
    student: dspy.LM
    teacher: Optional[dspy.LM]
    optimizer_metric: BaseMetric

    @classmethod
    def _get_constructor_parameters(cls, *, config: PromptTunerConfig) -> Dict[str, Any]:
        """
        Creates and configures the necessary components for DSPy prompt tuning.

        Args:
            config: Configuration containing all necessary parameters for the prompt tuner.
                   Must include student model config and optionally teacher model config.

        Returns:
            Dictionary of parameters needed to initialize the DSPyPromptTuner instance.
        """
        # Initialize student model and configure DSPy to use it
        student_model = DSPyLLMAdapter(llm_config=config.optimizer_config.student_config)
        dspy.configure(lm=student_model)

        # Initialize teacher model (or use student if not specified)
        if config.optimizer_config.teacher_config:
            teacher_model = DSPyLLMAdapter(llm_config=config.optimizer_config.teacher_config)
        else:
            teacher_model = student_model

        # Initialize metric for optimization
        optimizer_metric = BaseMetric.of(metric_config=config.optimizer_config.metric_config)

        return {
            "student": student_model,
            "teacher": teacher_model,
            "optimizer_metric": optimizer_metric,
            "config": config,
        }

    def tune(self, *, data: Dict[DatasetType, pd.DataFrame]) -> PromptTunerResult:
        """
        Tunes a prompt using the configured DSPy optimizer and training data.

        This method:
        1. Converts the input data to DSPy examples
        2. Creates a DSPy signature and module based on the prompt configuration
        3. Configures an evaluation function using the specified metric
        4. Applies the DSPy optimizer to generate an optimized prompt

        Args:
            data: DataFrame containing the training data with input and expected output fields

        Returns:
            The optimized prompt as a string

        Raises:
            ValueError: If the optimization process fails or returns invalid results
        """

        # Convert data to DSPy examples
        dataset: DspyDataset = DspyDataset(data=data, prompt_config=self.config.prompt_config)

        # Create signature and module separately
        signature: Type[Signature] = DSPyUtils.create_dspy_signature(
            prompt_config=self.config.prompt_config
        )
        module: Module = DSPyUtils.create_dspy_module(signature=signature)

        # Create evaluation function and evaluator
        evaluate: Callable = DSPyUtils.create_evaluation_function(metric=self.optimizer_metric)
        evaluator: dspy.Evaluate = Evaluate(
            devset=dataset.dev,
            metric=evaluate,
            num_threads=20,
            display_progress=True,
            max_errors=20,
            return_outputs=True,
        )

        optimizer = BaseDspyOptimizer.of(
            optimizerType=self.config.optimizer_config.type,
            student=self.student,
            teacher=self.teacher,
            module=module,
            evaluate=evaluate,
        )
        optimized_modules: List[dspy.Module] = optimizer.optimize(
            dataset=dataset, optimizer_params=self.config.optimizer_config.optimizer_params
        )

        tuned_prompts: List[TunedPrompt] = []
        for index, module in enumerate(optimized_modules):
            prompt: str = DSPyUtils.convert_module_to_prompt(module=module)
            validation_result: PromptEvaluationResult = DSPyUtils.evaluate_module(
                module=module, dataset=dataset.dev, evaluator=evaluator
            )
            test_result: PromptEvaluationResult = DSPyUtils.evaluate_module(
                module=module, dataset=dataset.test, evaluator=evaluator
            )
            tuned_prompt = TunedPrompt(
                prompt_id=str(index),
                prompt=prompt,
                validation_result=validation_result,
                test_result=test_result,
            )
            tuned_prompts.append(tuned_prompt)

        return PromptTunerResult(prompts=tuned_prompts)
