"""
Unified BERT Training Framework

A modular distributed training framework for BERT models supporting both 
regression and classification tasks with Ray Train, custom loss functions,
async S3 logging, and checkpoint resuming.
"""

from fmcore.framework.bert_training.config import (
    TaskConfig,
    ClassificationTaskConfig,
    RegressionTaskConfig,
    ProblemType
)
from fmcore.framework.bert_training.data_handler import DataHandler
from fmcore.framework.bert_training.losses import (
    CustomMSELoss,
    CustomCrossEntropyLoss,
    CustomBCEWithLogitsLoss,
    get_loss_function
)
from fmcore.framework.bert_training.model_handler import ModelHandler
from fmcore.framework.bert_training.trainer import CustomLossTrainer
from fmcore.framework.bert_training.logger import AsyncS3Logger, ValidationLogger
from fmcore.framework.bert_training.utils import generate_run_id, parse_s3_path
from fmcore.framework.bert_training.train_func import train_func

__all__ = [
    # Config
    "TaskConfig",
    "ClassificationTaskConfig",
    "RegressionTaskConfig",
    "ProblemType",
    # Data
    "DataHandler",
    # Losses
    "CustomMSELoss",
    "CustomCrossEntropyLoss",
    "CustomBCEWithLogitsLoss",
    "get_loss_function",
    # Model
    "ModelHandler",
    # Trainer
    "CustomLossTrainer",
    # Logging
    "AsyncS3Logger",
    "ValidationLogger",
    # Utils
    "generate_run_id",
    "parse_s3_path",
    # Training
    "train_func",
]

