"""
Binary classification evaluation metrics and visualizations.

This module provides functions to evaluate binary classification models using:
- Precision-Recall curves
- Reliability/Calibration curves
- Expected Calibration Error (ECE)

Works with output dataframes from ModelInference.predict() containing:
- Ground truth labels
- Predicted logits (pred_logits)
- Predicted scores/probabilities (pred_score)

Usage Example:
    import pandas as pd
    from fmcore.framework.bert_training.binary_evaluation import (
        plot_precision_recall_curve,
        plot_reliability_curve,
        calculate_expected_calibration_error
    )

    # Get predictions from ModelInference
    df = model_inference.predict(input_df)

    # Assuming df has columns: 'label', 'pred_score'
    # Plot precision-recall curve
    pr_plot = plot_precision_recall_curve(
        y_true=df['label'],
        y_score=df['pred_score'],
        title="Precision-Recall Curve"
    )

    # Plot reliability curve
    reliability_plot = plot_reliability_curve(
        y_true=df['label'],
        y_prob=df['pred_score'],
        n_bins=10,
        title="Reliability Diagram"
    )

    # Calculate ECE
    ece = calculate_expected_calibration_error(
        y_true=df['label'],
        y_prob=df['pred_score'],
        n_bins=10
    )
    print(f"Expected Calibration Error: {ece:.4f}")
"""

from typing import Union

import hvplot.pandas  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import auc, precision_recall_curve


# Set hvplot to use matplotlib backend
import hvplot

hvplot.extension("matplotlib")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str = "Precision-Recall Curve",
    width: int = 800,
    height: int = 600,
):
    """
    Plot precision-recall curve for binary classification.

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_score: Predicted probabilities for the positive class, shape (n_samples,)
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        hvplot object with precision-recall curve
    """
    # Calculate precision-recall curve
    # sklearn.metrics.precision_recall_curve returns:
    # - precision: array of precision values
    # - recall: array of recall values  
    # - thresholds: array of threshold values
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # Calculate area under PR curve
    pr_auc = auc(recall, precision)

    # Create dataframe for plotting
    df = pd.DataFrame({"recall": recall, "precision": precision})

    # Plot using hvplot with matplotlib backend
    plot = df.hvplot.line(
        x="recall",
        y="precision",
        title=f"{title} (AUC = {pr_auc:.4f})",
        xlabel="Recall",
        ylabel="Precision",
        width=width,
        height=height,
        color="blue",
        line_width=2,
        xlim=(0, 1),
        ylim=(0, 1),
        grid=True,
    )

    return plot


def plot_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    title: str = "Reliability Diagram (Calibration Curve)",
    width: int = 800,
    height: int = 600,
):
    """
    Plot reliability/calibration curve for binary classification.

    The reliability curve shows how well predicted probabilities match actual frequencies.
    A perfectly calibrated model will have points on the diagonal (y=x line).

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for the positive class, shape (n_samples,)
        n_bins: Number of bins to use for calibration curve
        strategy: Strategy for binning ('uniform' or 'quantile')
                 - 'uniform': bins have equal width
                 - 'quantile': bins have equal number of samples
        title: Plot title
        width: Plot width in pixels
        height: Plot height in pixels

    Returns:
        hvplot object with reliability curve
    """
    # Calculate calibration curve
    # sklearn.calibration.calibration_curve returns:
    # - prob_true: fraction of positives in each bin (actual frequency)
    # - prob_pred: mean predicted probability in each bin
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    # Create dataframe for plotting
    df = pd.DataFrame(
        {"mean_predicted_prob": prob_pred, "fraction_of_positives": prob_true}
    )

    # Plot calibration curve
    calibration_plot = df.hvplot.scatter(
        x="mean_predicted_prob",
        y="fraction_of_positives",
        title=title,
        xlabel="Mean Predicted Probability",
        ylabel="Fraction of Positives (Actual Frequency)",
        width=width,
        height=height,
        color="blue",
        size=100,
        xlim=(0, 1),
        ylim=(0, 1),
        grid=True,
        label="Model",
    )

    # Add diagonal line (perfectly calibrated)
    diagonal_df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    diagonal_plot = diagonal_df.hvplot.line(
        x="x",
        y="y",
        color="red",
        line_dash="dashed",
        line_width=2,
        label="Perfectly Calibrated",
    )

    # Combine plots
    combined_plot = calibration_plot * diagonal_plot

    return combined_plot


def calculate_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Calculate Expected Calibration Error (ECE) for binary classification.

    ECE measures the difference between predicted probabilities and actual frequencies
    across bins. Lower ECE indicates better calibration.

    ECE = sum_i (n_i / N) * |accuracy_i - confidence_i|

    where:
    - n_i: number of samples in bin i
    - N: total number of samples
    - accuracy_i: fraction of correct predictions in bin i
    - confidence_i: mean predicted probability in bin i

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for the positive class, shape (n_samples,)
        n_bins: Number of bins to use for ECE calculation
        strategy: Strategy for binning ('uniform' or 'quantile')

    Returns:
        Expected Calibration Error (float between 0 and 1)
    """
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # Create bins
    if strategy == "uniform":
        # Equal-width bins
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        # Equal-frequency bins (quantiles)
        bins = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicates
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'uniform' or 'quantile'")

    # Digitize predictions into bins
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Calculate ECE
    ece = 0.0
    n_total = len(y_true)

    for bin_idx in range(n_bins):
        # Get samples in this bin
        in_bin = bin_indices == bin_idx
        n_in_bin = np.sum(in_bin)

        if n_in_bin > 0:
            # Fraction of positives in bin (accuracy)
            accuracy = np.mean(y_true[in_bin])

            # Mean predicted probability in bin (confidence)
            confidence = np.mean(y_prob[in_bin])

            # Weighted absolute difference
            ece += (n_in_bin / n_total) * np.abs(accuracy - confidence)

    return float(ece)


def evaluate_binary_classification(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    plot_width: int = 800,
    plot_height: int = 600,
) -> dict:
    """
    Comprehensive evaluation of binary classification model.

    Generates all evaluation metrics and plots:
    - Precision-Recall curve
    - Reliability/Calibration curve
    - Expected Calibration Error

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_prob: Predicted probabilities for the positive class, shape (n_samples,)
        n_bins: Number of bins for calibration metrics
        strategy: Binning strategy ('uniform' or 'quantile')
        plot_width: Width of plots in pixels
        plot_height: Height of plots in pixels

    Returns:
        Dictionary containing:
        - 'pr_curve': hvplot object with precision-recall curve
        - 'reliability_curve': hvplot object with reliability curve
        - 'ece': Expected Calibration Error value
        - 'pr_auc': Area under precision-recall curve
    """
    # Calculate metrics
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_value = auc(recall, precision)
    ece_value = calculate_expected_calibration_error(
        y_true, y_prob, n_bins=n_bins, strategy=strategy
    )

    # Generate plots
    pr_plot = plot_precision_recall_curve(
        y_true, y_prob, width=plot_width, height=plot_height
    )
    reliability_plot = plot_reliability_curve(
        y_true, y_prob, n_bins=n_bins, strategy=strategy, width=plot_width, height=plot_height
    )

    return {
        "pr_curve": pr_plot,
        "reliability_curve": reliability_plot,
        "ece": ece_value,
        "pr_auc": pr_auc_value,
    }


# ==============================================================================
# Calibration Training Functions
# ==============================================================================


def train_binary_calibrator(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = "isotonic",
) -> Union[IsotonicRegression, Ridge]:
    """
    Train a calibrator for binary classification probabilities.

    Args:
        y_true: True binary labels (0 or 1), shape (n_samples,)
        y_score: Uncalibrated predicted probabilities, shape (n_samples,)
        method: Calibration method to use
               - 'isotonic': Isotonic regression (non-parametric, more flexible)
               - 'sigmoid' or 'platt': Platt scaling (parametric, assumes sigmoid shape)

    Returns:
        Trained calibrator model that can be used with .predict()

    Example:
        # Train calibrator on validation set
        calibrator = train_binary_calibrator(val_labels, val_probs, method='isotonic')

        # Apply to test set
        calibrated_probs = calibrator.predict(test_probs)
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score).reshape(-1, 1)

    if method == "isotonic":
        # Isotonic regression: non-parametric, preserves monotonicity
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_score.ravel(), y_true)
    elif method in ("sigmoid", "platt"):
        # Platt scaling: fits a logistic regression model
        # Ridge regression with very small regularization for numerical stability
        from scipy.special import logit

        # Convert probabilities to logits (inverse sigmoid)
        # Clip to avoid numerical issues
        y_score_clipped = np.clip(y_score, 1e-7, 1 - 1e-7)
        logits = logit(y_score_clipped)

        calibrator = Ridge(alpha=1e-6, fit_intercept=True)
        calibrator.fit(logits, y_true)
    else:
        raise ValueError(
            f"Unknown calibration method: {method}. "
            f"Use 'isotonic' or 'sigmoid'/'platt'"
        )

    return calibrator


def train_multiclass_calibrator(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    method: str = "isotonic",
) -> list:
    """
    Train per-class calibrators for multi-class classification.

    For each class, trains a binary calibrator that calibrates the probability
    for that specific class.

    Args:
        y_true: True class labels (0 to n_classes-1), shape (n_samples,)
        y_probs: Uncalibrated class probabilities, shape (n_samples, n_classes)
        method: Calibration method ('isotonic' or 'sigmoid')

    Returns:
        List of calibrators (one per class) that can be applied to each class probability

    Example:
        # Train calibrators on validation set
        calibrators = train_multiclass_calibrator(
            val_labels,
            val_probs,  # shape: (n_samples, n_classes)
            method='isotonic'
        )

        # Apply to test set
        calibrated_probs = apply_multiclass_calibrator(test_probs, calibrators)
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    if y_probs.ndim == 1:
        raise ValueError(
            "y_probs must be 2D (n_samples, n_classes) for multiclass calibration"
        )

    n_classes = y_probs.shape[1]
    calibrators = []

    # Train one calibrator per class
    for class_idx in range(n_classes):
        # Create binary labels: 1 if true class == class_idx, else 0
        y_binary = (y_true == class_idx).astype(int)

        # Get probabilities for this class
        y_score_class = y_probs[:, class_idx]

        # Train calibrator for this class
        calibrator = train_binary_calibrator(y_binary, y_score_class, method=method)
        calibrators.append(calibrator)

    return calibrators


def apply_multiclass_calibrator(
    y_probs: np.ndarray,
    calibrators: list,
    normalize: bool = True,
) -> np.ndarray:
    """
    Apply per-class calibrators to multiclass probabilities.

    Args:
        y_probs: Uncalibrated class probabilities, shape (n_samples, n_classes)
        calibrators: List of trained calibrators (one per class)
        normalize: Whether to renormalize probabilities to sum to 1

    Returns:
        Calibrated class probabilities, shape (n_samples, n_classes)
    """
    y_probs = np.array(y_probs)
    n_classes = y_probs.shape[1]

    if len(calibrators) != n_classes:
        raise ValueError(
            f"Number of calibrators ({len(calibrators)}) must match "
            f"number of classes ({n_classes})"
        )

    calibrated_probs = np.zeros_like(y_probs)

    # Apply each calibrator to its corresponding class probabilities
    for class_idx, calibrator in enumerate(calibrators):
        class_probs = y_probs[:, class_idx]

        if isinstance(calibrator, IsotonicRegression):
            calibrated_probs[:, class_idx] = calibrator.predict(class_probs)
        elif isinstance(calibrator, Ridge):
            # For Platt scaling, need to apply sigmoid after prediction
            from scipy.special import logit, expit

            class_probs_clipped = np.clip(class_probs, 1e-7, 1 - 1e-7)
            logits = logit(class_probs_clipped).reshape(-1, 1)
            calibrated_logits = calibrator.predict(logits)
            calibrated_probs[:, class_idx] = expit(calibrated_logits)

    # Renormalize to ensure probabilities sum to 1
    if normalize:
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        calibrated_probs = calibrated_probs / row_sums

    return calibrated_probs


def train_regression_calibrator(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    method: str = "isotonic",
) -> Union[IsotonicRegression, Ridge]:
    """
    Train a calibrator for regression predictions.

    Useful when model predictions are systematically biased (over/under-predicting).

    Args:
        y_true: True target values, shape (n_samples,)
        y_pred: Uncalibrated predictions, shape (n_samples,)
        method: Calibration method
               - 'isotonic': Isotonic regression (non-parametric)
               - 'linear': Linear calibration (Ridge regression with minimal regularization)

    Returns:
        Trained calibrator that maps uncalibrated predictions to calibrated ones

    Example:
        # Train calibrator on validation set
        calibrator = train_regression_calibrator(
            val_targets,
            val_predictions,
            method='isotonic'
        )

        # Apply to test set
        calibrated_preds = calibrator.predict(test_predictions)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred).reshape(-1, 1)

    if method == "isotonic":
        # Isotonic regression
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_pred.ravel(), y_true)
    elif method == "linear":
        # Linear calibration using Ridge regression
        calibrator = Ridge(alpha=1e-6, fit_intercept=True)
        calibrator.fit(y_pred, y_true)
    else:
        raise ValueError(
            f"Unknown calibration method for regression: {method}. "
            f"Use 'isotonic' or 'linear'"
        )

    return calibrator

