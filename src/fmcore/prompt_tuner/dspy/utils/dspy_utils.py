import dspy
import pandas as pd

from langchain_core.prompts.chat import ChatPromptTemplate
from typing import Callable, Dict, List

from fmcore.prompt_tuner.dspy.datasets.base_dataset import DspyDataset
from fmcore.prompt_tuner.evaluator import BaseEvaluator
from fmcore.prompt_tuner.evaluator.types.evaluator_types import EvaluatorConfig
from fmcore.prompt_tuner.types.prompt_tuner_types import PromptConfig, PromptEvaluationResult
from fmcore.types.enums.dataset_enums import DatasetType


class DSPyUtils:
    """
    Utility class for working with DSPy in prompt tuning and optimization tasks.

    This class provides static methods for setting up and configuring various components
    of the DSPy framework, including optimizers, datasets, signatures, modules, and evaluation
    functions. It serves as a bridge between the fmcore configuration objects and DSPy's
    expected interfaces.
    """

    @staticmethod
    def create_dspy_dataset(
        data: Dict[DatasetType, pd.DataFrame], prompt_config: PromptConfig
    ) -> DspyDataset:
        """
        Creates a DSPy dataset from a DataFrame and prompt configuration.

        Wraps the input data in a DspyDataset instance which handles the mapping
        between DataFrame columns and DSPy input/output fields as specified in the
        prompt configuration.

        Args:
            data: DataFrame containing the dataset to be used for training or evaluation
            prompt_config: Configuration containing information about input and output fields

        Returns:
            A configured DspyDataset instance ready for use with DSPy components
        """
        return DspyDataset(data=data, prompt_config=prompt_config)

    @staticmethod
    def create_dspy_signature(prompt_config: PromptConfig) -> dspy.Signature:
        """
        Creates a DSPy Signature based on the prompt configuration.

        Dynamically generates a DSPy Signature class with input and output fields
        as defined in the prompt configuration. The signature is used to define
        the interface that DSPy modules will use for processing.

        Args:
            prompt_config: Configuration containing input and output fields and prompt text

        Returns:
            A dynamically created DSPy Signature class with the appropriate fields
        """
        # Create a DSPy Signature class dictionary with annotations
        attrs = {
            "__annotations__": {},
            # Use prompt text as class docstring if available
            "__doc__": prompt_config.prompt if prompt_config.prompt else "",
        }

        # Dynamically add input fields with their type annotations
        for field in prompt_config.input_fields:
            # Use provided field type or default to str if not available
            field_type = getattr(field, "type", str)
            attrs["__annotations__"][field.name] = field_type
            attrs[field.name] = dspy.InputField(desc=field.description)

        # Dynamically add output fields with their type annotations
        for field in prompt_config.output_fields:
            field_type = getattr(field, "type", str)
            attrs["__annotations__"][field.name] = field_type
            attrs[field.name] = dspy.OutputField(desc=field.description)

        # Create the Signature class dynamically with type annotations
        TaskSignature = type("TaskSignature", (dspy.Signature,), attrs)

        return TaskSignature

    @staticmethod
    def create_dspy_module(signature: dspy.Signature) -> dspy.Module:
        """
        Creates a DSPy Module that uses the provided signature.

        Generates a module that uses Chain of Thought reasoning with the
        specified signature to process inputs and generate outputs.

        Args:
            signature: The DSPy Signature that defines the input/output interface

        Returns:
            An instantiated DSPy Module configured with the provided signature
        """

        class TaskModule(dspy.Module):
            """
            A DSPy module that uses Chain of Thought reasoning for prediction.

            This module wraps a ChainOfThought predictor with the specified signature
            to provide a simple forward interface for making predictions.
            """

            def __init__(self, signature: dspy.Signature):
                """
                Initialize the TaskModule with the provided signature.

                Args:
                    signature: The DSPy Signature defining input and output fields
                """
                super().__init__()
                self.signature = signature
                # Use Chain of Thought for enhanced reasoning capabilities
                self.predictor = dspy.ChainOfThought(signature=self.signature)

            def forward(self, **kwargs):
                """
                Process inputs and generate predictions using the ChainOfThought predictor.

                Args:
                    **kwargs: Input field values matching the signature's input fields

                Returns:
                    A DSPy Prediction object containing the generated outputs
                """
                prediction = self.predictor(**kwargs)
                return prediction

        # Return an instantiated TaskModule ready for use
        return TaskModule(signature=signature)

    @staticmethod
    def create_dspy_signature_from_prompt_config(prompt_config: PromptConfig) -> dspy.Module:
        """
        Create a DSPy module from a given prompt configuration.

        Args:
            prompt_config (PromptConfig): The prompt configuration used to generate the DSPy signature.

        Returns:
            dspy.Module: A DSPy module created from the generated signature.
        """
        signature: dspy.Signature = DSPyUtils.create_dspy_signature(prompt_config=prompt_config)
        return DSPyUtils.create_dspy_module(signature=signature)

    @staticmethod
    def create_evaluation_function_from_evaluator(evaluator: BaseEvaluator) -> Callable:
        def evaluate_func(example: dspy.Example, prediction: dspy.Prediction, trace=None):
            """
            Evaluates a single example-prediction pair using the configured metric.

            Args:
                example: The DSPy example containing input data
                prediction: The model's prediction to evaluate
                trace: Optional trace information from DSPy (not used)

            Returns:
                Evaluation score as determined by the configured criteria
            """
            # Prepare the data structure expected by the metric
            row = {
                "input": example.toDict(),
                "output": prediction.toDict(),
            }

            return evaluator.evaluate(data=row)

        return evaluate_func

    @staticmethod
    def create_dspy_evaluate_from_evaluator_config(evaluator_config: EvaluatorConfig) -> Callable:
        """
        Create a DSPy evaluation function from a given evaluator configuration.

        Args:
            evaluator_config (EvaluatorConfig): The evaluator configuration used to initialize the evaluator.

        Returns:
            Callable: A function that performs evaluation using the configured evaluator.
        """
        evaluator = BaseEvaluator.of(evaluator_config=evaluator_config)
        return DSPyUtils.create_evaluation_function_from_evaluator(evaluator=evaluator)

    @staticmethod
    def convert_module_to_messages(module: dspy.Module) -> List[Dict[str, str]]:
        """
        Converts a DSPy module to a list of chat messages.

        This method takes a DSPy module and converts it into a list of chat messages
        that can be used with language models. It extracts the signature and demos
        from the module and formats them using DSPy's ChatAdapter.

        Args:
            module: The DSPy module to convert

        Returns:
            A list of dictionaries representing chat messages, where each dictionary
            contains 'role' and 'content' keys
        """
        # Create chat adapter to handle message formatting
        adapter = dspy.ChatAdapter()

        # Get input field names from signature and create template variables
        signature: dspy.Signature = module.signature
        inputs = {field_name: f"{{{field_name}}}" for field_name in signature.input_fields.keys()}

        # Format the module into chat messages using the adapter
        messages = adapter.format(signature=signature, demos=module.demos, inputs=inputs)

        return messages

    @staticmethod
    def convert_module_to_prompt(module: dspy.Module) -> str:
        """
        Converts a DSPy module to a single prompt string.

        This method takes a DSPy module and converts it into a single prompt string by
        extracting the content from chat messages and concatenating them. It first converts
        the module to chat messages using convert_module_to_messages() and then extracts
        just the content fields.

        Args:
            module: The DSPy module to convert

        Returns:
            A string containing the concatenated content from all chat messages
        """
        # First get the messages using the existing method
        messages = DSPyUtils.convert_module_to_messages(module)
        prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(messages=messages)
        keys: List[str] = list(module.signature.input_fields.keys()) + list(
            module.signature.output_fields.keys()
        )
        prompt = prompt_template.format(**{key: "{{{}}}".format(key) for key in keys})
        return prompt

    @staticmethod
    def evaluate_module(
        module: dspy.Module, dataset: List[dspy.Example], evaluator: dspy.Evaluate
    ) -> PromptEvaluationResult:
        """
        Evaluates a DSPy module using a dataset and an evaluation metric.

        Args:
            module: The DSPy module to evaluate.
            dataset: The dataset used for evaluation.
            evaluator: The evaluation metric.

        Returns:
            A PromptEvaluationResult containing the evaluation score and processed results.
        """
        score, evaluation_results = evaluator(module, devset=dataset)

        prompt_template = ChatPromptTemplate.from_messages(
            messages=DSPyUtils.convert_module_to_messages(module)
        )

        processed_results = []
        for example, prediction, is_correct in evaluation_results:
            record = {**example.inputs().toDict(), **prediction.toDict()}
            prompt = prompt_template.format(**record)
            row = {**record, "prompt": prompt, "is_correct": is_correct}
            processed_results.append(row)

        return PromptEvaluationResult(score=score, data=pd.DataFrame(processed_results))
