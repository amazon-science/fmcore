"""
Configuration models for unified BERT training framework.

Uses Pydantic BaseModel for type checking and validation.
All configs are frozen (immutable) after creation.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, conint, validator


class ProblemType(str, Enum):
    """Enum for problem types."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class TaskConfig(BaseModel):
    """
    Base configuration for training tasks.

    All fields are required by the user - no defaults for critical parameters.
    """

    # Model configuration
    model_name: str  # e.g., "microsoft/deberta-v3-large"
    tokenizer_name: str  # Usually same as model_name
    max_length: int  # e.g., 512

    # Data paths and columns
    train_data_path: str  # S3 path to train parquet file
    val_data_path: str  # S3 path to validation parquet file
    text_columns: List[str]  # e.g., ["search_query", "product_text"]
    label_column: str  # e.g., "ecvrm" or "label"

    # Training configuration
    problem_type: ProblemType
    learning_rate: float
    num_train_steps: int  # Use steps instead of epochs for better control
    per_device_batch_size: int
    gradient_accumulation_steps: int
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float

    # Evaluation and checkpointing
    eval_steps: int  # How often to evaluate
    save_steps: int  # How often to save checkpoints
    save_total_limit: Optional[int] = (
        None  # Max number of checkpoints to keep (None = keep all)
    )
    # If save_total_limit is set:
    # - load_best_model_at_end=False (default): Keeps the N most recent checkpoints
    # - load_best_model_at_end=True: Keeps the N best checkpoints based on metric_for_best_model
    load_best_model_at_end: bool = False  # Whether to load best model at end
    metric_for_best_model: Optional[str] = (
        None  # Metric to use for best model selection (e.g., "eval_loss", "eval_f1")
    )

    # Distributed training (Ray)
    num_workers: int  # Number of Ray workers (GPUs)

    # S3 paths
    s3_output_base_path: str  # Base S3 path for this training run
    checkpoint_resume_path: Optional[str] = (
        None  # S3 path to checkpoint dir to resume from
    )

    # Logging
    log_every_n_steps: int = 25  # How often to log metrics to console and S3

    # Logging verbosity: 0=silent (only initial setup), 1=standard (default), 2=detailed (log every step)
    verbosity: conint(ge=0, le=2) = 1

    # Hardware
    dataloader_num_workers: int = 4  # Number of dataloader worker processes

    class Config:
        frozen = True  # Make instances immutable

    @staticmethod
    def get_run_id() -> str:
        """
        Generate unique run ID based on UTC timestamp.

        Format: YYYY-MM-DDTHH-MM-SSZ (filesystem-safe, uses hyphens instead of colons)

        Returns:
            ISO timestamp string (e.g., "2024-11-24T14-30-45Z")
        """
        return datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")


class ClassificationTaskConfig(TaskConfig):
    """
    Configuration for classification tasks.

    Note: num_classes is NOT specified here - it will be auto-detected from the data.
    Loss type (BCE vs CrossEntropy) will be auto-selected based on num_classes.
    """

    @validator("problem_type")
    def check_problem_type(cls, v):
        """Ensure problem_type is set to classification."""
        if v != ProblemType.CLASSIFICATION:
            raise ValueError(
                f"ClassificationTaskConfig requires problem_type='classification', "
                f"got '{v}'"
            )
        return v


class RegressionTaskConfig(TaskConfig):
    """
    Configuration for regression tasks.
    """

    # Optional label scaling for numerical stability
    label_scale_factor: float = 1.0

    @validator("problem_type")
    def check_problem_type(cls, v):
        """Ensure problem_type is set to regression."""
        if v != ProblemType.REGRESSION:
            raise ValueError(
                f"RegressionTaskConfig requires problem_type='regression', got '{v}'"
            )
        return v

    @validator("label_scale_factor")
    def check_scale_factor(cls, v):
        """Ensure label_scale_factor is positive."""
        if v <= 0:
            raise ValueError(f"label_scale_factor must be positive, got {v}")
        return v
