"""
Custom loss functions for BERT training.

Implements custom loss functions (DO NOT use HuggingFace built-ins):
- CustomMSELoss: For regression tasks
- CustomCrossEntropyLoss: For multi-class classification
- CustomBCEWithLogitsLoss: For binary classification
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fmcore.framework.bert_training.config import ProblemType, TaskConfig


class CustomMSELoss(nn.Module):
    """Mean Squared Error Loss for regression tasks."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss.

        Args:
            logits: Model predictions, shape (batch_size, 1) or (batch_size,)
            labels: Ground truth, shape (batch_size,)

        Returns:
            MSE loss (scalar tensor)
        """
        logits = logits.squeeze()
        labels = labels.squeeze()
        return F.mse_loss(logits, labels)


class CustomCrossEntropyLoss(nn.Module):
    """Cross-Entropy Loss for multi-class classification (num_classes > 2)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-entropy loss.

        Args:
            logits: Model predictions, shape (batch_size, num_classes)
            labels: Ground truth class indices, shape (batch_size,)

        Returns:
            Cross-entropy loss (scalar tensor)
        """
        return F.cross_entropy(logits, labels)


class CustomBCEWithLogitsLoss(nn.Module):
    """Binary Cross-Entropy with Logits Loss for binary classification (num_classes = 2)."""

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute BCE with logits loss.

        Args:
            logits: Model predictions, shape (batch_size, 1) or (batch_size,)
            labels: Ground truth binary labels (0 or 1), shape (batch_size,)

        Returns:
            BCE with logits loss (scalar tensor)
        """
        logits = logits.squeeze()
        labels = labels.float()
        return F.binary_cross_entropy_with_logits(logits, labels)


def get_loss_function(
    config: TaskConfig, num_classes: Optional[int] = None
) -> nn.Module:
    """
    Factory function to get appropriate loss function based on task type.

    Args:
        config: Task configuration
        num_classes: Number of classes (required for classification tasks)

    Returns:
        Loss function module (nn.Module instance)

    Raises:
        ValueError: If num_classes is not provided for classification or
                   if problem_type is unknown
    """
    if config.problem_type == ProblemType.REGRESSION:
        return CustomMSELoss()

    elif config.problem_type == ProblemType.CLASSIFICATION:
        if num_classes is None:
            raise ValueError("num_classes required for classification tasks")

        if num_classes == 2:
            # Binary classification: use BCE with logits
            return CustomBCEWithLogitsLoss()
        else:
            # Multi-class classification: use Cross-Entropy
            return CustomCrossEntropyLoss(num_classes)

    else:
        raise ValueError(f"Unknown problem_type: {config.problem_type}")
