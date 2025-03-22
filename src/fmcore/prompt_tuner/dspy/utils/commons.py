from typing import Callable, Dict, List, Type, Union
from asteval import Interpreter
from pandas import DataFrame
import dspy
from dspy.teleprompt import Teleprompter
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.teleprompt.bootstrap import BootstrapFewShot
from fmcore.metrics.base_metric import BaseMetric
from fmcore.prompt_tuner.dspy.datasets.base_dataset import DspyDataset
from fmcore.types.enums.metric_enums import (
    EvaluationFieldType,
    MetricFramework,
)
from fmcore.types.metric_types import MetricResult
from fmcore.types.prompt_tuner_types import OptimizerConfig, PromptConfig
from fmcore.types.enums.prompt_tuner_enums import DspyOptimizerType


class DSPyUtils:
    """
    Utility class for working with DSPy in prompt tuning and optimization tasks.

    This class provides static methods for setting up and configuring various components
    of the DSPy framework, including optimizers, datasets, signatures, modules, and evaluation
    functions. It serves as a bridge between the fmcore configuration objects and DSPy's
    expected interfaces.
    """

    @staticmethod
    def get_optimizer(
        student: dspy.LM,
        teacher: dspy.LM,
        optimzer_config: OptimizerConfig,
        evaluate_func: Callable,
    ) -> Teleprompter:
        """
        Creates and configures a DSPy optimizer based on configuration parameters.

        Acts as a factory method that instantiates the appropriate optimizer type
        based on the configuration. Currently supports MIPROv2 with plans to expand
        to other optimizer types.

        Args:
            student: The language model that will be used for the actual task (student model)
            teacher: The language model used for prompt optimization (teacher model)
            optimzer_config: Configuration for the optimizer including type and parameters
            evaluate_func: Evaluation function used to assess prediction quality

        Returns:
            A configured DSPy Teleprompter instance that can be used for prompt optimization

        Note:
            Currently only MIPROv2 is implemented. Future implementations will support
            additional optimizer types based on DspyOptimizerType.
        """
        # Factory method that returns the appropriate optimizer based on the optimizer type
        # TODO: Extend to support multiple optimizer types from DspyOptimizerType enum
        optimizer: Teleprompter = MIPROv2(
            prompt_model=teacher,
            task_model=student,
            metric=evaluate_func,
            **optimzer_config.params,
        )

        return optimizer

    @staticmethod
    def create_dspy_dataset(data: DataFrame, prompt_config: PromptConfig) -> DspyDataset:
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
    def create_dspy_signature(prompt_config: PromptConfig) -> Type[dspy.Signature]:
        """
        Creates a DSPy Signature based on the prompt configuration.

        Dynamically generates a DSPy Signature class with input and output fields
        as defined in the prompt configuration. The signature is used to define
        the interface that DSPy modules will use for processing.

        Args:
            prompt_config: Configuration containing input and output fields and prompt text

        Returns:
            A dynamically created DSPy Signature class with the appropriate fields

        Example:
            >>> signature = DSPyUtils.create_dspy_signature(prompt_config)
            >>> module = DSPyUtils.create_dspy_module(signature)
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
    def create_dspy_module(signature: Type[dspy.Signature]) -> Type[dspy.Module]:
        """
        Creates a DSPy Module that uses the provided signature.

        Generates a module that uses Chain of Thought reasoning with the
        specified signature to process inputs and generate outputs.

        Args:
            signature: The DSPy Signature that defines the input/output interface

        Returns:
            An instantiated DSPy Module configured with the provided signature

        Example:
            >>> signature = DSPyUtils.create_dspy_signature(prompt_config)
            >>> module = DSPyUtils.create_dspy_module(signature)
            >>> result = module(input_field1="value1", input_field2="value2")
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
    def create_evaluation_function(metric: BaseMetric) -> Callable:
        """
        Creates an evaluation function that uses the configured metric.

        The function evaluates DSPy predictions by applying the metric and interpreting
        the criteria expression to determine the quality of the prediction. The created
        function follows the interface expected by DSPy optimizers.

        Args:
            metric: The configured metric object used to evaluate predictions

        Returns:
            A callable function that takes an example and prediction and returns a
            numerical or boolean evaluation score

        Example:
            >>> metric = SomeMetric(config=metric_config)
            >>> eval_func = DSPyUtils.create_evaluation_function(metric)
        """
        # Store criteria expression once to avoid re-fetching it in each evaluation call
        criteria = metric.config.metric_params["criteria"]
        # Create a base interpreter for expression evaluation

        def evaluate_func(
            example: dspy.Example, prediction: dspy.Prediction, trace=None
        ) -> Union[float, bool]:
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
                EvaluationFieldType.INPUT.name: example.toDict(),
                EvaluationFieldType.OUTPUT.name: prediction.toDict(),
            }

            # Apply the metric to get evaluation results
            metric_result: MetricResult = metric.evaluate(data=row)
            evaluation_response = metric_result.model_dump(exclude_none=True)

            # Evaluate the criteria expression to get the final score
            expression_evaluator = Interpreter()
            expression_evaluator.symtable.update(evaluation_response)
            return expression_evaluator(criteria)

        return evaluate_func

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
        inputs = {
            field_name: f"{{{field_name}}}" for field_name in signature.input_fields.keys()
        }

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

        # Extract content from each message and join with newlines
        prompt = "".join(message["content"] for message in messages)

        return prompt
