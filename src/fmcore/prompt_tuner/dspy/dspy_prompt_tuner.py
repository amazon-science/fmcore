import dspy
import pandas as pd
from typing import Dict, List

from fmcore.prompt_tuner.base_prompt_tuner import BasePromptTuner
from fmcore.prompt_tuner.dspy.datasets.base_dataset import DspyDataset
from fmcore.prompt_tuner.dspy.optimizer_wrapper.base_dspy_optimizer_wrapper import BaseDspyOptimizerWrapper
from fmcore.prompt_tuner.dspy.utils.dspy_utils import DSPyUtils
from fmcore.prompt_tuner.types.enums.prompt_tuner_enums import PromptTunerFramework
from fmcore.prompt_tuner.types.prompt_tuner_types import (
    PromptTunerResult,
    PromptEvaluationResult,
    TunedPrompt,
)
from fmcore.types.enums.dataset_enums import DatasetType
from fmcore.prompt_tuner.dspy.chat_adapters.chat_adapter import custom_prepare_instructions

# Override the DSPy chat adapter's `prepare_instructions` function
# This is a workaround to replace DSPy's default instruction preparation mechanism with our custom implementation.
# It's essentially a hack to modify the behavior of the DSPy library to suit our specific needs.
import dspy.adapters.chat_adapter as chat_adapter_module

chat_adapter_module.prepare_instructions = custom_prepare_instructions


class DSPyPromptTuner(BasePromptTuner):
    """
    A prompt tuner implementation using the DSPy framework.

    This class provides functionality to optimize prompts using DSPy optimizers such as MIPROv2
    or BootstrapFewShot. It uses a student model to generate responses and evaluates them using
    a specified metric to iteratively improve the prompt.

    Attributes:
        aliases (list): List of prompt tuner framework aliases, including DSPY.
    """

    aliases = [PromptTunerFramework.DSPY]

    def tune_with_data(self, *, data: Dict[DatasetType, pd.DataFrame]) -> PromptTunerResult:
        """
        Tunes a prompt using the DSPy optimizer and the provided training data.

        This method performs the following steps:
        1. Converts the input data into DSPy dataset examples.
        2. Creates a DSPy signature and module, which serve as the foundation for optimization.
        3. Uses the DSPy optimizer to optimize the dataset and generate optimized modules.
        4. Evaluates each optimized module using a specified evaluation metric on both validation and test datasets.
        5. Converts the optimized modules into prompts and stores the evaluation results.
        6. Returns a collection of tuned prompts sorted by test score in descending order.

        Args:
            data (Dict[DatasetType, pd.DataFrame]): A dictionary mapping dataset types (such as training, validation, and test)
                                                     to their respective pandas DataFrames. The data should include both input
                                                     features and expected output labels for training.

        Returns:
            PromptTunerResult: An object containing a list of tuned prompts, each with associated validation and test results.

        Raises:
            ValueError: If the optimization process fails or returns invalid results.
        """
        # Step 1: Convert data into DSPy dataset examples
        dataset: DspyDataset = DspyDataset(data=data, prompt_config=self.config.prompt_config)

        # Step 2: Initialize DSPy optimizer
        optimizer_wrapper = BaseDspyOptimizerWrapper.of(prompt_tuner_config=self.config)
        optimized_modules: List[dspy.Module] = optimizer_wrapper.optimize(dataset=dataset)

        # Step 3: Configure the evaluation function
        # DSPy Evaluate natively handles parallelization for module evaluation
        # Pinning it to 20 threads for now to avoid resource contention while calling LLMs
        evaluator = dspy.Evaluate(
            devset=dataset.dev,
            metric=optimizer_wrapper.evaluate,
            num_threads=20,  # Enable parallel evaluation with 20 threads
            display_progress=True,  # Show progress during evaluation
            max_errors=20,  # Limit the number of errors to avoid excessive failures
            return_outputs=True,  # Ensure evaluation outputs are returned for further analysis
        )

        # Step 4: Iterate over optimized modules to create tuned prompts
        tuned_prompts: List[TunedPrompt] = []
        for index, module in enumerate(optimized_modules):
            # Convert each module to a text prompt
            # For multimodal prompt tuners, a list of messages would be a better representation.
            # Currently, we support returning text prompts, but support for messages will be added in a future update (TODO).
            prompt: str = DSPyUtils.convert_module_to_prompt(module=module)

            # Evaluate the module for validation and test datasets
            validation_result: PromptEvaluationResult = DSPyUtils.evaluate_module(
                module=module, dataset=dataset.dev, evaluator=evaluator
            )
            test_result: PromptEvaluationResult = DSPyUtils.evaluate_module(
                module=module, dataset=dataset.test, evaluator=evaluator
            )

            # Create a TunedPrompt object
            tuned_prompt = TunedPrompt(
                prompt_id=str(index),
                prompt=prompt,
                validation_result=validation_result,
                test_result=test_result,
            )
            tuned_prompts.append(tuned_prompt)

        # Sort tuned prompts by descending validation score for better results
        tuned_prompts.sort(key=lambda prompt: prompt.test_result.score, reverse=True)

        return PromptTunerResult(prompts=tuned_prompts)
