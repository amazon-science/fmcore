"""
Training function that runs on each Ray worker.

This module contains the core training logic that executes on each distributed worker.
"""

import os
import tempfile
import time
import traceback

import numpy as np
import ray.train
import ray.train.huggingface.transformers
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoTokenizer, TrainerCallback, TrainingArguments

from fmcore.framework.bert_training.config import (
    ClassificationTaskConfig,
    ProblemType,
    RegressionTaskConfig,
)
from fmcore.framework.bert_training.data_handler import DataHandler
from fmcore.framework.bert_training.logger import AsyncS3Logger, ValidationLogger
from fmcore.framework.bert_training.losses import get_loss_function
from fmcore.framework.bert_training.model_handler import ModelHandler
from fmcore.framework.bert_training.trainer import CustomLossTrainer


def format_time_duration(seconds: float) -> str:
    """
    Format a time duration in seconds to a human-readable string.

    Args:
        seconds: Time duration in seconds

    Returns:
        Formatted string (e.g., "45.2s", "3.5m", "1.2h")
    """
    if seconds >= 3600:  # >= 1 hour
        return f"{seconds / 3600:.1f}h"
    elif seconds >= 60:  # >= 1 minute
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds:.1f}s"


def compute_regression_metrics(
    predictions: np.ndarray, labels: np.ndarray, is_rank_0: bool
) -> dict:
    """
    Compute comprehensive regression metrics.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        is_rank_0: Whether this is rank 0 (for printing)

    Returns:
        Dictionary of metrics
    """
    try:
        # Comprehensive regression metrics
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, predictions)
        r2 = r2_score(labels, predictions)

        # Adjusted R²
        n = len(labels)
        p = 1  # number of predictors (1 for single output)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((labels - predictions) / (labels + 1e-10))) * 100

        # Median Absolute Error
        median_ae = np.median(np.abs(labels - predictions))

        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "median_ae": float(median_ae),
            "r2": float(r2),
            "adj_r2": float(adj_r2),
            "mape": float(mape),
        }

        # Note: Metrics table is printed in on_evaluate callback with runtime stats
        return metrics

    except Exception as e:
        if is_rank_0:
            print(f"[ERROR] compute_regression_metrics failed: {e}")
            traceback.print_exc()
        raise e


def compute_binary_classification_metrics(
    predictions: np.ndarray, labels: np.ndarray, is_rank_0: bool
) -> dict:
    """
    Compute comprehensive binary classification metrics.

    Args:
        predictions: Model prediction probabilities
        labels: Ground truth labels (0 or 1)
        is_rank_0: Whether this is rank 0 (for printing)

    Returns:
        Dictionary of metrics
    """
    try:
        # Get predicted classes
        predictions_class = (predictions > 0.5).astype(int)

        # Basic metrics
        accuracy = accuracy_score(labels, predictions_class)
        precision = precision_score(labels, predictions_class, zero_division=0)
        recall = recall_score(labels, predictions_class, zero_division=0)
        f1 = f1_score(labels, predictions_class, zero_division=0)

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(labels, predictions)
        except Exception:
            roc_auc = 0.0

        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(labels, predictions_class).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "specificity": float(specificity),
            "ppv": float(ppv),
            "npv": float(npv),
        }

        # Note: Comprehensive metrics table (including runtime stats) is printed in on_evaluate callback
        return metrics

    except Exception as e:
        if is_rank_0:
            print(f"[ERROR] compute_binary_classification_metrics failed: {e}")
            traceback.print_exc()
        raise e


def compute_multiclass_classification_metrics(
    predictions: np.ndarray, labels: np.ndarray, is_rank_0: bool
) -> dict:
    """
    Compute comprehensive multi-class classification metrics.

    Args:
        predictions: Model prediction logits
        labels: Ground truth labels
        is_rank_0: Whether this is rank 0 (for printing)

    Returns:
        Dictionary of metrics
    """
    try:
        # Get probabilities and predicted classes
        probs = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)
        predictions_class = np.argmax(predictions, axis=-1)

        # Basic metrics
        accuracy = accuracy_score(labels, predictions_class)

        # Precision with different averaging
        precision_macro = precision_score(
            labels, predictions_class, average="macro", zero_division=0
        )
        precision_micro = precision_score(
            labels, predictions_class, average="micro", zero_division=0
        )
        precision_weighted = precision_score(
            labels, predictions_class, average="weighted", zero_division=0
        )

        # Recall with different averaging
        recall_macro = recall_score(
            labels, predictions_class, average="macro", zero_division=0
        )
        recall_micro = recall_score(
            labels, predictions_class, average="micro", zero_division=0
        )
        recall_weighted = recall_score(
            labels, predictions_class, average="weighted", zero_division=0
        )

        # F1 with different averaging
        f1_macro = f1_score(labels, predictions_class, average="macro", zero_division=0)
        f1_micro = f1_score(labels, predictions_class, average="micro", zero_division=0)
        f1_weighted = f1_score(
            labels, predictions_class, average="weighted", zero_division=0
        )

        # ROC-AUC (One-vs-Rest)
        try:
            roc_auc_ovr = roc_auc_score(
                labels, probs, multi_class="ovr", average="macro"
            )
        except Exception:
            roc_auc_ovr = 0.0

        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "precision_micro": float(precision_micro),
            "precision_weighted": float(precision_weighted),
            "recall_macro": float(recall_macro),
            "recall_micro": float(recall_micro),
            "recall_weighted": float(recall_weighted),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "f1_weighted": float(f1_weighted),
            "roc_auc_ovr_macro": float(roc_auc_ovr),
        }

        # Note: Comprehensive metrics table (including runtime stats) is printed in on_evaluate callback
        return metrics

    except Exception as e:
        if is_rank_0:
            print(f"[ERROR] compute_multiclass_classification_metrics failed: {e}")
            traceback.print_exc()
        raise e


def train_func(config_dict: dict):
    """
    Training function that runs on each Ray worker.

    This function is executed on each Ray worker and handles:
    - Data loading and tokenization
    - Model initialization
    - Training loop with custom loss
    - Async logging to S3
    - Validation result logging

    Args:
        config_dict: Dictionary containing:
            - All TaskConfig fields (serialized)
            - 'run_id': Unique run identifier (ISO timestamp)
    """
    # Get Ray context
    train_context = ray.train.get_context()
    rank = train_context.get_world_rank()
    is_rank_0 = rank == 0

    # Extract run_id
    run_id = config_dict.pop("run_id")

    # Reconstruct config from dict
    if config_dict["problem_type"] == "classification":
        config = ClassificationTaskConfig(**config_dict)
    elif config_dict["problem_type"] == "regression":
        config = RegressionTaskConfig(**config_dict)
    else:
        raise NotImplementedError(
            f"Invalid problem type: {config_dict['problem_type']}"
        )

    # Print status (rank 0 only)
    if is_rank_0:
        print("=" * 80)
        print("[SETUP] Starting distributed training")
        print(f"  Run ID: {run_id}")
        print(f"  Workers: {train_context.get_world_size()}")
        print(f"  Problem Type: {config.problem_type}")
        print(f"  Model: {config.model_name}")
        print("=" * 80)

    # Create temp dir for HF Trainer checkpoints
    temp_output_dir = tempfile.mkdtemp(prefix="hf_trainer_")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name,
        use_fast=False,  # For DeBERTa compatibility
    )

    # Load datasets
    data_handler = DataHandler(config, tokenizer, is_rank_0)
    train_dataset, val_dataset = data_handler.load_and_prepare_datasets()
    num_classes = data_handler.num_classes  # None for regression

    if is_rank_0:
        print(
            f"[DATA] Loaded {len(train_dataset):,} train samples, {len(val_dataset):,} val samples"
        )
        if num_classes is not None:
            print(f"[DATA] Auto-detected {num_classes} classes")

    # Initialize model
    model = ModelHandler.initialize_model(config, num_classes)

    # Get loss function
    loss_fn = get_loss_function(config, num_classes)

    if is_rank_0:
        print(f"[MODEL] Loaded {config.model_name}")
        print(f"[LOSS] Using {loss_fn.__class__.__name__}")

    # Storage for predictions to be accessed by callback
    # This allows us to save predictions without calling predict() again
    last_eval_predictions = {"predictions": None, "labels": None, "step": None}

    # Compute metrics function
    def compute_metrics(eval_pred):
        """
        Compute comprehensive metrics for evaluation.

        NOTE: This function is called by Trainer after evaluation completes.
        In distributed training, predictions are automatically gathered to rank 0
        before this function is called, so we can safely save them here.
        """
        try:
            logits, labels = eval_pred
            predictions = logits.squeeze()

            # Store predictions for the callback to save to S3
            # In DDP, compute_metrics is called only on rank 0 with gathered predictions
            # Store the raw arrays (they're already numpy arrays from Trainer)
            if is_rank_0:
                # logits and labels from eval_pred are already numpy arrays
                last_eval_predictions["predictions"] = (
                    predictions.copy() if hasattr(predictions, "copy") else predictions
                )
                last_eval_predictions["labels"] = (
                    labels.copy() if hasattr(labels, "copy") else labels
                )

            # Compute metrics based on task type
            if config.problem_type == ProblemType.REGRESSION:
                # Descale if needed
                if (
                    hasattr(config, "label_scale_factor")
                    and config.label_scale_factor != 1.0
                ):
                    predictions = predictions / config.label_scale_factor
                    labels = labels / config.label_scale_factor

                # Filter out NaN/Inf
                valid_mask = ~(
                    np.isnan(predictions)
                    | np.isinf(predictions)
                    | np.isnan(labels)
                    | np.isinf(labels)
                )

                if np.sum(valid_mask) == 0:
                    return {"mse": float("inf"), "mae": float("inf")}

                predictions = predictions[valid_mask]
                labels = labels[valid_mask]

                return compute_regression_metrics(predictions, labels, is_rank_0)

            else:  # Classification
                if num_classes == 2:
                    # Binary classification
                    return compute_binary_classification_metrics(
                        predictions, labels, is_rank_0
                    )
                else:
                    # Multi-class classification
                    return compute_multiclass_classification_metrics(
                        predictions, labels, is_rank_0
                    )

        except Exception as e:
            if is_rank_0:
                print(f"[ERROR] compute_metrics failed: {e}")
                traceback.print_exc()
            raise e

    # HuggingFace TrainingArguments
    training_args = TrainingArguments(
        output_dir=temp_output_dir,
        # Training schedule
        max_steps=config.num_train_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        # Batch sizes
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        # Regularization
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        # Logging
        # CRITICAL: logging_steps MUST be the same for all workers to avoid DDP deadlock!
        # HuggingFace Trainer has implicit synchronization at logging steps.
        # We control output via is_rank_0 checks in the callback, not here.
        logging_steps=config.log_every_n_steps,  # SAME for all workers
        disable_tqdm=True,  # Disable tqdm on ALL workers
        report_to="none",
        log_on_each_node=False,
        logging_first_step=False,
        # Control which processes log
        # This suppresses the raw dict logs {'loss': ...} from non-global-rank-0 workers
        log_level="info",  # Main process (global rank 0) logs at INFO level
        log_level_replica="error",  # Replicas (all other ranks) only log errors
        # Checkpointing
        save_only_model=True,
        save_total_limit=config.save_total_limit,  # Max checkpoints to keep (None = keep all)
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,  # Used if load_best_model_at_end=True
    )

    # Create Custom Trainer
    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        custom_loss_fn=loss_fn,
    )

    # Add Ray Train callback
    ray_callback = ray.train.huggingface.transformers.RayTrainReportCallback()
    trainer.add_callback(ray_callback)

    # Initialize async S3 logger
    s3_logger = AsyncS3Logger(
        s3_base_path=config.s3_output_base_path, run_id=run_id, rank=rank
    )

    # Initialize validation logger (rank 0 only)
    val_logger = None
    if is_rank_0:
        val_logger = ValidationLogger(
            s3_base_path=config.s3_output_base_path, run_id=run_id
        )

    # Custom callback for detailed progress tracking and async logging
    class DetailedProgressCallback(TrainerCallback):
        """
        Callback for detailed progress tracking and async logging.

        Provides clear visibility into training progress:
        - Step-by-step progress updates (controlled by verbosity)
        - Evaluation start/end with timing
        - Checkpoint save start/end with timing
        - Validation predictions logging status

        Verbosity levels:
        - 0: Silent (no training output - for background jobs; only shows initial setup)
        - 1: Standard (default - steps shown with other logs)
        - 2: Detailed (every step logged separately)
        """

        def __init__(self, verbosity: int = 1):
            self.verbosity = verbosity
            self.current_step = 0
            self.last_log_time = None
            self.training_start_time = None

        def on_step_end(self, args, state, control, **kwargs):
            """Log progress after each training step (rank 0 only)."""
            try:
                if is_rank_0:
                    # Initialize training start time on first step
                    if self.training_start_time is None:
                        self.training_start_time = time.time()

                    # Track current step
                    self.current_step = state.global_step

                    # Only print every step if verbosity >= 2
                    if self.verbosity >= 2:
                        progress = (state.global_step / state.max_steps) * 100

                        # Calculate elapsed time and ETA
                        elapsed = time.time() - self.training_start_time
                        steps_per_sec = (
                            state.global_step / elapsed if elapsed > 0 else 0
                        )
                        remaining_steps = state.max_steps - state.global_step
                        eta = (
                            remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                        )

                        print(
                            f"[STEP {state.global_step}/{state.max_steps}] "
                            f"({progress:.1f}% complete) | "
                            f"elapsed: {format_time_duration(elapsed)}, "
                            f"ETA: {format_time_duration(eta)}"
                        )
            except Exception as e:
                if is_rank_0 and self.verbosity >= 1:
                    print(f"[ERROR] on_step_end failed: {e}")
                raise e

        def on_log(self, args, state, control, logs=None, **kwargs):
            """
            Log metrics to S3 asynchronously and print summary.

            Each worker logs to its own S3 file. The identical loss values
            across workers prove that gradient synchronization is working.
            """
            try:
                if logs is not None:
                    # Log to S3 asynchronously (each worker to its own file)
                    # The logger has timeout protection to prevent hangs
                    metrics = {"step": state.global_step, **logs}
                    s3_logger.log(metrics)

                # Print training metrics summary (rank 0 only, if verbosity >= 1)
                if (
                    is_rank_0
                    and self.verbosity >= 1
                    and logs is not None
                    and "loss" in logs
                ):
                    # Initialize training start time if not set
                    if self.training_start_time is None:
                        self.training_start_time = time.time()

                    lr = logs.get("learning_rate", 0)
                    loss = logs.get("loss", 0)

                    # Calculate elapsed time and ETA
                    elapsed = time.time() - self.training_start_time
                    steps_per_sec = state.global_step / elapsed if elapsed > 0 else 0
                    remaining_steps = state.max_steps - state.global_step
                    eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

                    # For verbosity == 1, include step info since we don't print it separately
                    if self.verbosity == 1:
                        progress = (state.global_step / state.max_steps) * 100
                        print(
                            f"[STEP {state.global_step}/{state.max_steps}] ({progress:.1f}% complete) | "
                            f"elapsed: {format_time_duration(elapsed)}, ETA: {format_time_duration(eta)} | "
                            f"loss={loss:.6f}, lr={lr:.2e}"
                        )
                    else:
                        # For verbosity >= 2, step is already printed, just print metrics
                        print(
                            f"[TRAIN] Step {state.global_step}: loss={loss:.6f}, lr={lr:.2e}"
                        )

            except Exception as e:
                if is_rank_0 and self.verbosity >= 1:
                    print(f"[ERROR] on_log failed at step {state.global_step}: {e}")
                    traceback.print_exc()
                raise e

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            """Handle evaluation and save predictions/metrics to S3."""
            try:
                if is_rank_0 and self.verbosity >= 1:
                    print(f"\n{'=' * 80}")
                    print(
                        f"[EVAL START] Step {state.global_step}: Starting evaluation..."
                    )
                    eval_start = time.time()

                # Evaluation happens here (handled by Trainer)

                if is_rank_0 and metrics is not None and val_logger is not None:
                    if self.verbosity >= 1:
                        eval_time = time.time() - eval_start
                        print(
                            f"[EVAL COMPLETE] Evaluation finished in {eval_time:.2f} seconds"
                        )

                        # Print comprehensive metrics table with ALL fields
                        # NOTE: ALWAYS include all metrics - classification metrics, loss, runtime, throughput, epoch
                        print(f"\n{'=' * 80}")
                        if config.problem_type == ProblemType.CLASSIFICATION:
                            if num_classes == 2:
                                print("BINARY CLASSIFICATION METRICS:")
                            else:
                                print("MULTI-CLASS CLASSIFICATION METRICS:")
                        else:
                            print("REGRESSION METRICS:")

                        # Format all metrics in a vertical table (metric name | value)
                        import pandas as pd

                        # Convert metrics dict to vertical format: two columns (Metric, Value)
                        metrics_df = pd.DataFrame(
                            list(metrics.items()), columns=["Metric", "Value"]
                        )
                        try:
                            print(metrics_df.to_markdown(index=False, floatfmt=".6f"))
                        except ImportError:
                            # Fallback if tabulate not installed
                            print(
                                metrics_df.to_string(
                                    index=False, float_format=lambda x: f"{x:.6f}"
                                )
                            )
                        print(f"{'=' * 80}\n")

                    # Save predictions and metrics to S3
                    # NOTE: We do NOT call trainer.predict() here because:
                    # 1. Evaluation just completed (that's why on_evaluate was called)
                    # 2. Calling predict() again would trigger another full evaluation pass
                    # 3. In distributed training, this causes synchronization issues/hangs
                    #
                    # Instead, we use predictions that were saved in compute_metrics().
                    # compute_metrics() is called with predictions already gathered to rank 0.
                    if self.verbosity >= 1:
                        print(
                            "[SAVING] Saving validation predictions and metrics to S3..."
                        )
                        print(
                            f"  Location: {os.path.join(config.s3_output_base_path, run_id, 'validation')}"
                        )
                    save_start = time.time()

                    try:
                        # Get predictions that were saved in compute_metrics
                        # These are already gathered to rank 0 and converted to numpy arrays
                        predictions = last_eval_predictions.get("predictions")
                        labels = last_eval_predictions.get("labels")

                        val_logger.log_validation_results(
                            predictions=predictions,
                            labels=labels,
                            metrics=metrics,
                            step=state.global_step,
                        )

                        if self.verbosity >= 1:
                            save_time = time.time() - save_start
                            print(
                                f"[SAVED] Validation results saved to S3 in {save_time:.2f} seconds"
                            )
                            if predictions is not None:
                                print(
                                    f"  Files:\n  predictions_step={state.global_step:08d}.parquet\n  metrics_step={state.global_step:08d}.json"
                                )
                            else:
                                print(
                                    f"  Files: metrics_step={state.global_step:08d}.json (predictions not available)"
                                )

                    except Exception as e:
                        if self.verbosity >= 1:
                            print(f"[WARNING] Failed to save validation metrics: {e}")
                            traceback.print_exc()

                    if self.verbosity >= 1:
                        print(f"{'=' * 80}\n")

            except Exception as e:
                if is_rank_0 and self.verbosity >= 1:
                    print(
                        f"[ERROR] on_evaluate failed at step {state.global_step}: {e}"
                    )
                    traceback.print_exc()
                raise e

        def on_save(self, args, state, control, **kwargs):
            """Log when checkpoint saving starts."""
            try:
                if is_rank_0 and self.verbosity >= 1:
                    print(f"\n{'=' * 80}")
                    print(
                        f"[CHECKPOINT START] Step {state.global_step}: Saving checkpoint to S3..."
                    )
                    self.checkpoint_start = time.time()
            except Exception as e:
                if is_rank_0 and self.verbosity >= 1:
                    print(f"[ERROR] on_save failed: {e}")
                    traceback.print_exc()
                raise e

        def on_step_begin(self, args, state, control, **kwargs):
            """Check if checkpoint save completed (called at next step)."""
            try:
                if (
                    is_rank_0
                    and self.verbosity >= 1
                    and hasattr(self, "checkpoint_start")
                ):
                    checkpoint_time = time.time() - self.checkpoint_start
                    print(
                        f"[CHECKPOINT COMPLETE] Checkpoint saved in {checkpoint_time:.2f} seconds"
                    )
                    print(
                        f"  Location: {os.path.join(config.s3_output_base_path, run_id, 'checkpoints', f'checkpoint-{state.global_step}')}"
                    )
                    print(f"{'=' * 80}\n")
                    delattr(self, "checkpoint_start")
            except Exception as e:
                if is_rank_0 and self.verbosity >= 1:
                    print(f"[ERROR] on_step_begin failed: {e}")
                    traceback.print_exc()
                raise e

    trainer.add_callback(DetailedProgressCallback(verbosity=config.verbosity))

    # Prepare trainer for Ray Train
    trainer = ray.train.huggingface.transformers.prepare_trainer(trainer)

    # Start training (with optional checkpoint resuming)
    resume_from_checkpoint = (
        config.checkpoint_resume_path if config.checkpoint_resume_path else None
    )

    if is_rank_0:
        if resume_from_checkpoint:
            print(f"[CHECKPOINT] Resuming from: {resume_from_checkpoint}")
        print("=" * 80)
        print("[TRAINING] Starting training loop...")
        print(f"  Logging every: {config.log_every_n_steps} steps")
        print(f"  Evaluation every: {config.eval_steps} steps")
        print(f"  Checkpoint every: {config.save_steps} steps")
        print("=" * 80)

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception as e:
        if is_rank_0:
            print(f"[ERROR] Training failed: {e}")
            traceback.print_exc()
        raise e

    # Shutdown logger
    s3_logger.shutdown()

    if is_rank_0:
        print("=" * 80)
        print("[COMPLETE] Training finished successfully")
        print(f"  Run ID: {run_id}")
        print(f"  Logs: {config.s3_output_base_path}/{run_id}/logs/")
        print(f"  Validation: {config.s3_output_base_path}/{run_id}/validation/")
        print(f"  Checkpoints: {config.s3_output_base_path}/{run_id}/checkpoints/")
        print("=" * 80)
