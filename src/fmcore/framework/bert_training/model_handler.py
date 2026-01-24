"""
Model initialization handler for BERT models.

Handles initialization of BERT models with appropriate output heads
for both regression and classification tasks.
"""

from typing import Optional

from transformers import AutoModelForSequenceClassification

from fmcore.framework.bert_training.config import ProblemType, TaskConfig


class ModelHandler:
    """Handler for initializing BERT models with task-specific heads."""

    @staticmethod
    def initialize_model(
        config: TaskConfig, num_classes: Optional[int] = None
    ) -> AutoModelForSequenceClassification:
        """
        Initialize BERT model with appropriate output head for the task.

        Args:
            config: Task configuration
            num_classes: Number of classes (required for classification tasks)

        Returns:
            Initialized model (AutoModelForSequenceClassification)

        Raises:
            ValueError: If num_classes is not provided for classification or
                       if problem_type is unknown

        Notes:
            - For regression: num_labels=1
            - For binary classification: num_labels=1 (single logit + BCE loss)
            - For multi-class: num_labels=num_classes (logits for each class + CE loss)
            - We don't set problem_type in the model (we handle loss ourselves)
        """
        if config.problem_type == ProblemType.REGRESSION:
            # Regression: num_labels=1, no problem_type (we use custom loss)
            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=1,
            )

        elif config.problem_type == ProblemType.CLASSIFICATION:
            if num_classes is None:
                raise ValueError("num_classes required for classification tasks")

            # Classification: num_labels depends on binary vs multi-class
            # For binary (num_classes=2), we use single output + BCE loss
            # For multi-class (num_classes>2), we use num_classes outputs + CE loss
            num_labels = 1 if num_classes == 2 else num_classes

            model = AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=num_labels,
            )

        else:
            raise ValueError(f"Unknown problem_type: {config.problem_type}")

        return model
