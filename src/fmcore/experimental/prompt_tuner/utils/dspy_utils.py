from typing import Callable, Dict, List, Type
from pandas import DataFrame
import dspy
from dspy.teleprompt import Teleprompter
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.teleprompt.bootstrap import BootstrapFewShot
from fmcore.experimental.metrics.base_metric import BaseMetric
from fmcore.experimental.types.enums.metric_enums import EvaluationFieldType, MetricFramework
from fmcore.experimental.types.metric_types import MetricResult
from fmcore.experimental.types.prompt_tuner_types import OptimizerConfig, PromptConfig
from fmcore.experimental.types.enums.prompt_tuner_enums import DspyOptimizerType


class DSPyUtils:

    @staticmethod
    def get_optimizer(
        student: dspy.LM,
        teacher: dspy.LM,
        optimzer_config: OptimizerConfig,
        evaluate_func: Callable,
    ) -> Teleprompter:
        
        # This will be a factory method that returns the appropriate optimizer based on the optimizer type
        # TODO: Add more optimizers
        optimizer: Teleprompter = MIPROv2(
                prompt_model=teacher,
                task_model=student,
                metric=evaluate_func,
                **optimzer_config.params,
            )
    
        return optimizer

    @staticmethod
    def create_dspy_signature(prompt_config: PromptConfig) -> Type[dspy.Signature]:
        """
        Creates a DSPy Signature based on the prompt configuration.

        Args:
            prompt_config: Configuration containing input and output fields

        Returns:
            A DSPy Signature class with the appropriate fields
        """

        # Create a DSPy Signature class dictionary with annotations
        attrs = {
            "__annotations__": {},
            "__doc__": prompt_config.prompt if prompt_config.prompt else "",
        }

        # Dynamically add input and output fields with type annotations
        for field in prompt_config.input_fields:
            # Assume field has a type attribute, otherwise default to str
            field_type = getattr(field, "type", str)
            attrs["__annotations__"][field.name] = field_type
            attrs[field.name] = dspy.InputField(desc=field.description)

        for field in prompt_config.output_fields:
            field_type = getattr(field, "type", str)
            attrs["__annotations__"][field.name] = field_type
            attrs[field.name] = dspy.OutputField(desc=field.description)

        # Create the class dynamically with type annotations
        TaskSignature = type("TaskSignature", (dspy.Signature,), attrs)

        return TaskSignature

    @staticmethod
    def create_dspy_module(signature: Type[dspy.Signature]) -> Type[dspy.Module]:
        """
        Creates a DSPy Module that uses the provided signature.

        Args:
            signature: The DSPy Signature to use in the module

        Returns:
            A DSPy Module class configured with the signature
        """

        class TaskModule(dspy.Module):
            def __init__(self, signature: dspy.Signature):
                super().__init__()
                self.signature = signature
                self.predictor = dspy.ChainOfThought(signature=self.signature)

            def forward(self, **kwargs):
                prediction = self.predictor(**kwargs)
                return prediction

        return TaskModule(signature=signature)

    @staticmethod
    def convert_to_dspy_examples(
        data: DataFrame, prompt_config: PromptConfig
    ) -> List[dspy.Example]:
        """
        Converts pandas DataFrame records to DSPy Example objects.

        Args:
            data: DataFrame containing the training data
            prompt_config: Configuration containing input and output fields

        Returns:
            List of DSPy Example objects
        """
        from dspy.datasets import DataLoader

        loader = DataLoader()
        input_keys = [field.name for field in prompt_config.input_fields]
        examples = loader.from_pandas(data, fields=data.columns.tolist(), input_keys=input_keys)

        return examples

    @staticmethod
    def evaluate(
        example: dspy.Example, prediction: dspy.Prediction, metric: BaseMetric
    ) -> MetricResult:
        row = {
            EvaluationFieldType.INPUT.name: example.toDict(),
            EvaluationFieldType.OUTPUT.name: prediction.toDict(),
        }

        metric_result: MetricResult = metric.evaluate(data=row)
        return metric_result.model_dump(exclude_none=True)