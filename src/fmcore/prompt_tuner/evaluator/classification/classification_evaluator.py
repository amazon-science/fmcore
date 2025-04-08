from typing import Dict

from fmcore.mapper.equality_checker_mapper import EqualityCheckerMapper
from fmcore.prompt_tuner.evaluator.classification.classification_evaluator_types import ClassificationParams
from fmcore.prompt_tuner.evaluator.types.evaluator_types import EvaluatorConfig
from fmcore.prompt_tuner.evaluator.base_evaluator import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator[Dict, bool]):
    """
    Evaluator that checks if the predicted label matches the ground truth.

    This evaluator uses `EqualityCheckerMapper` to compare values extracted from the input dictionary
    using dot-paths. It supports nested structures and returns a boolean result indicating whether
    the prediction is correct.
    """

    aliases = ["CLASSIFICATION"]

    equality_checker: EqualityCheckerMapper

    @classmethod
    def _get_instance(cls, *, evaluator_config: EvaluatorConfig) -> "ClassificationEvaluator":
        """
        Initializes a ClassificationEvaluator from the given configuration.

        Args:
            evaluator_config (EvaluatorConfig): Configuration containing evaluation parameters.

        Returns:
            LLMAsJudgeBooleanEvaluator: An evaluator instance with required mappers configured.
        """
        classification_params: ClassificationParams = evaluator_config.evaluator_params

        equality_checker: EqualityCheckerMapper = EqualityCheckerMapper(
            prediction_key=classification_params.prediction_field,
            ground_truth_key=classification_params.ground_truth_field,
        )

        return ClassificationEvaluator(config=evaluator_config, equality_checker=equality_checker)

    def evaluate(self, data: Dict) -> bool:
        """
        Synchronously evaluates if prediction matches the ground truth.

        Args:
            data (Dict): Input dictionary containing prediction and ground truth.

        Returns:
            bool: True if prediction equals ground truth, else False.
        """
        return self.equality_checker.map(data)

    async def aevaluate(self, data: Dict) -> bool:
        """
        Asynchronously evaluates if prediction matches the ground truth.

        Args:
            data (Dict): Input dictionary containing prediction and ground truth.

        Returns:
            bool: True if prediction equals ground truth, else False.
        """
        return await self.equality_checker.amap(data)
