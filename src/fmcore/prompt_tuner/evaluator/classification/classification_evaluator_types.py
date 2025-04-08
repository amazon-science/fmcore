from fmcore.prompt_tuner.evaluator import BaseEvaluatorParams


class ClassificationParams(BaseEvaluatorParams):
    """
    Configuration parameters for the Classification evaluator.

    This evaluator checks whether the response matches the expected ground truth
    based on a given prompt and evaluation criteria.

    Attributes:
        ground_truth_field (str): The key in the input data that holds the expected label.
        prediction_field (str): The key in the input data that holds the predicted label.
    """

    aliases = ["CLASSIFICATION"]

    prediction_field: str
    ground_truth_field: str
