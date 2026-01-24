"""
Inference module for trained BERT models.

This module handles loading trained models from S3 checkpoints and running
inference on pandas DataFrames with appropriate post-processing for:
- Binary classification (logits + sigmoid scores)
- Multi-class classification (per-class logits + softmax scores)
- Regression (scaled + descaled predictions)

Usage Example:
    from fmcore.framework.bert_training.config import ClassificationTaskConfig, ProblemType
    from fmcore.framework.bert_training.inference import ModelInferenceHandler

    # Define same config used during training
    config = ClassificationTaskConfig(
        model_name="microsoft/deberta-v3-large",
        tokenizer_name="microsoft/deberta-v3-large",
        max_length=512,
        text_columns=["search_query", "product_text"],
        label_column="label",
        problem_type=ProblemType.CLASSIFICATION,
        # ... other fields not needed for inference (can use dummy values)
        train_data_path="dummy",
        val_data_path="dummy",
        learning_rate=1e-5,
        num_train_steps=1000,
        per_device_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_steps=100,
        save_steps=100,
        num_workers=1,
        s3_output_base_path="dummy",
    )

    # Initialize inference handler (without calibration)
    handler = ModelInference(
        checkpoint_s3_path="s3://bucket/path/to/checkpoint-1234/",
        task_config=config,
        device="cuda:0",  # or None for auto-detect
        batch_size=32,
        cache_dir="/path/to/cache",  # Optional: cache checkpoints locally
        verbosity=1  # 0=silent, 1=model loading (default), 2=all details
    )

    # Run inference
    df_with_predictions = handler.predict(input_df)
    
    # Optional: Train calibrator on validation set
    from fmcore.framework.bert_training.binary_evaluation import train_binary_calibrator
    
    # Get validation predictions
    val_df = handler.predict(val_input_df)
    
    # Train calibrator
    calibrator = train_binary_calibrator(
        y_true=val_df['true_label'],
        y_score=val_df['pred_score'],
        method='isotonic'  # or 'sigmoid'
    )
    
    # Create new handler with calibrator
    handler_calibrated = ModelInference(
        checkpoint_s3_path="s3://bucket/path/to/checkpoint-1234/",
        task_config=config,
        device="cuda:0",
        batch_size=32,
        cache_dir="/path/to/cache",
        calibrator=calibrator  # Pass trained calibrator
    )
    
    # Predictions will now include calibrated scores
    df_calibrated = handler_calibrated.predict(test_df)
"""

import atexit
import json
import os
import shutil
import tempfile
from typing import Dict, Optional

import boto3
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fmcore.framework.bert_training.config import ProblemType, TaskConfig
from fmcore.framework.bert_training.utils import parse_s3_path


class ModelInference:
    """
    Handler for loading trained BERT models and running inference.

    Supports:
    - Automatic checkpoint download from S3
    - GPU/CPU device placement
    - Batched inference for large dataframes
    - Task-specific post-processing (binary, multi-class, regression)
    """

    def __init__(
        self,
        checkpoint_s3_path: str,
        task_config: TaskConfig,
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        verbosity: int = 1,
        calibrator=None,
    ):
        """
        Initialize inference handler.

        Args:
            checkpoint_s3_path: S3 path to checkpoint directory
                               (e.g., "s3://bucket/path/checkpoint-1234/")
            task_config: TaskConfig used during training (must have same
                        text_columns, max_length, tokenizer_name, problem_type)
            device: Device to run inference on (e.g., "cuda:0", "cpu").
                   If None, auto-detects (uses GPU if available).
            batch_size: Number of samples to process per batch
            cache_dir: Optional local directory to cache downloaded checkpoints.
                      If provided, checkpoints are downloaded here and reused.
                      If None, checkpoints are downloaded to a temporary directory
                      that is cleaned up on exit.
            verbosity: Logging verbosity level (0=silent, 1=model loading info, 2=all details)
                      - 0: Silent except for errors/warnings
                      - 1: Show model loading progress (default)
                      - 2: Show all details including predict() progress
            calibrator: Optional calibrator for post-processing predictions
                       - Binary classification: single calibrator (IsotonicRegression or Ridge)
                       - Multi-class classification: list of calibrators (one per class)
                       - Regression: single calibrator (IsotonicRegression or Ridge)
                       Train using fmcore.framework.bert_training.binary_evaluation.train_*_calibrator()

        Raises:
            ValueError: If checkpoint doesn't exist or config is invalid
        """
        self.checkpoint_s3_path = checkpoint_s3_path.rstrip("/")
        self.config = task_config
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.use_temp_dir = cache_dir is None
        self.verbosity = verbosity
        self.calibrator = calibrator

        # Validate config
        if self.config.problem_type not in [
            ProblemType.CLASSIFICATION,
            ProblemType.REGRESSION,
        ]:
            raise ValueError(f"Invalid problem_type: {self.config.problem_type}")

        if self.verbosity >= 1:
            print("[INFERENCE] Initializing inference handler")
            print(f"  Checkpoint: {self.checkpoint_s3_path}")
            print(f"  Problem type: {self.config.problem_type}")

        if self.verbosity >= 2:
            print(f"  Batch size: {self.batch_size}")

        # Download checkpoint (or use cached version)
        if self.verbosity >= 1:
            if self.use_temp_dir:
                print("[INFERENCE] Downloading checkpoint from S3...")
            else:
                print(
                    "[INFERENCE] Loading checkpoint from cache (or downloading if not cached)..."
                )
        self.local_checkpoint_dir = self._download_checkpoint(self.checkpoint_s3_path)

        if self.verbosity >= 2:
            print(f"  Local directory: {self.local_checkpoint_dir}")

        # Register cleanup handler (only for temp directories)
        if self.use_temp_dir:
            atexit.register(self._cleanup)

        # Load config.json to auto-detect num_labels
        config_path = os.path.join(self.local_checkpoint_dir, "config.json")
        with open(config_path, "r") as f:
            model_config = json.load(f)

        # Get num_labels: either from config directly or infer from id2label
        # This follows HuggingFace's standard practice for handling configs
        if "num_labels" in model_config:
            num_labels = model_config["num_labels"]
        elif "id2label" in model_config:
            # Infer num_labels from id2label dictionary
            num_labels = len(model_config["id2label"])
        else:
            raise ValueError(
                "Cannot determine num_labels from config.json. "
                "Config must have either 'num_labels' or 'id2label' field."
            )

        # Determine num_classes from num_labels
        if self.config.problem_type == ProblemType.REGRESSION:
            self.num_classes = None
            if self.verbosity >= 1:
                print(f"  Task: Regression (num_labels={num_labels})")
        elif self.config.problem_type == ProblemType.CLASSIFICATION:
            if num_labels == 1:
                # Binary classification
                self.num_classes = 2
                if self.verbosity >= 1:
                    print(f"  Task: Binary classification (num_labels={num_labels})")
            else:
                # Multi-class classification
                self.num_classes = num_labels
                if self.verbosity >= 1:
                    print(
                        f"  Task: Multi-class classification (num_classes={self.num_classes})"
                    )
        else:
            raise ValueError(f"Unknown problem_type: {self.config.problem_type}")

        # Load tokenizer
        # Try to load from checkpoint first (if custom tokenizer was saved)
        # Otherwise fall back to original pre-trained tokenizer
        if self.verbosity >= 1:
            print("[INFERENCE] Loading tokenizer...")

        # Check if tokenizer files exist in checkpoint
        # Common tokenizer files: tokenizer_config.json, vocab.json, spm.model, etc.
        tokenizer_files = [
            "tokenizer_config.json",
            "vocab.json",
            "spm.model",
            "tokenizer.json",
        ]
        has_tokenizer = any(
            os.path.exists(os.path.join(self.local_checkpoint_dir, f))
            for f in tokenizer_files
        )

        if has_tokenizer:
            # Load custom tokenizer from checkpoint
            if self.verbosity >= 2:
                print(
                    "  Found tokenizer files in checkpoint, loading custom tokenizer..."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_checkpoint_dir,
                use_fast=False,  # For DeBERTa compatibility
            )
        else:
            # Load original pre-trained tokenizer from HuggingFace Hub
            if self.verbosity >= 2:
                print(
                    "  No tokenizer files in checkpoint, loading from HuggingFace Hub..."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name,
                use_fast=False,  # For DeBERTa compatibility
            )

        if self.verbosity >= 1:
            print(f"  Tokenizer loaded: {self.tokenizer.__class__.__name__}")

        # Load model
        if self.verbosity >= 1:
            print("[INFERENCE] Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.local_checkpoint_dir
        )
        if self.verbosity >= 1:
            print(f"  Model loaded: {self.model.__class__.__name__}")

        # Set device
        if device is None:
            # Auto-detect: use GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.verbosity >= 1:
                print(f"  Device: {self.device} (auto-detected)")
        else:
            try:
                self.device = torch.device(device)
                if self.verbosity >= 1:
                    print(f"  Device: {self.device} (user-specified)")
            except Exception as e:
                if self.verbosity >= 1:
                    print(f"  WARNING: Invalid device '{device}', falling back to CPU")
                    print(f"  Error: {e}")
                self.device = torch.device("cpu")
                if self.verbosity >= 1:
                    print(f"  Device: {self.device} (fallback)")

        # Move model to device
        try:
            self.model.to(self.device)
            if self.verbosity >= 2:
                print(f"  Model moved to {self.device}")
        except Exception as e:
            if self.verbosity >= 1:
                print(f"  WARNING: Failed to move model to {self.device}, using CPU")
                print(f"  Error: {e}")
            self.device = torch.device("cpu")
            self.model.to(self.device)
            if self.verbosity >= 2:
                print(f"  Model moved to {self.device} (fallback)")

        # Set model to eval mode
        self.model.eval()
        if self.verbosity >= 1:
            print("[INFERENCE] Initialization complete")

    def _download_checkpoint(self, s3_path: str) -> str:
        """
        Download checkpoint files from S3 to local directory.

        If cache_dir is provided, downloads to cache and reuses on subsequent calls.
        If cache_dir is None, downloads to a temporary directory.

        Args:
            s3_path: S3 path to checkpoint directory

        Returns:
            Path to local directory containing checkpoint files

        Raises:
            ValueError: If checkpoint directory doesn't exist or is empty
        """
        # Parse S3 path
        bucket, key_prefix = parse_s3_path(s3_path)

        # Ensure key_prefix ends with "/" for directory listing
        if len(key_prefix) > 0 and not key_prefix.endswith("/"):
            key_prefix += "/"

        # Determine local directory
        if self.cache_dir is not None:
            # Use cache directory - create a subdirectory based on S3 path
            # to avoid conflicts between different checkpoints
            # Use bucket + key_prefix as unique identifier
            cache_subdir = f"{bucket}/{key_prefix}".replace("/", "_").rstrip("_")
            local_dir = os.path.join(self.cache_dir, cache_subdir)

            # Check if already cached
            if os.path.exists(local_dir):
                # Verify it has files
                if len(os.listdir(local_dir)) > 0:
                    if self.verbosity >= 2:
                        print("    Using cached checkpoint (skipping download)")
                    return local_dir
                else:
                    # Empty directory, download
                    if self.verbosity >= 2:
                        print("    Cache directory exists but is empty, downloading...")
            else:
                # Create cache directory
                os.makedirs(local_dir, exist_ok=True)
        else:
            # Use temporary directory
            local_dir = tempfile.mkdtemp(prefix="model_checkpoint_")

        # Initialize S3 client
        s3_client = boto3.client("s3")

        try:
            # List all files in the checkpoint directory
            paginator = s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=key_prefix)

            files_to_download = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        s3_key = obj["Key"]
                        # Skip directories (keys ending with /)
                        if not s3_key.endswith("/"):
                            # Get filename relative to prefix
                            filename = s3_key[len(key_prefix) :]
                            if filename:  # Skip empty filenames
                                files_to_download.append((s3_key, filename))

            if len(files_to_download) == 0:
                raise ValueError(
                    f"No files found in checkpoint directory: s3://{bucket}/{key_prefix}"
                )

            if self.verbosity >= 2:
                print(f"    Found {len(files_to_download)} files to download")

            # Download all files
            for s3_key, filename in files_to_download:
                local_path = os.path.join(local_dir, filename)

                # Create subdirectories if needed
                local_subdir = os.path.dirname(local_path)
                if local_subdir and not os.path.exists(local_subdir):
                    os.makedirs(local_subdir, exist_ok=True)

                # Download file
                s3_client.download_file(bucket, s3_key, local_path)

            if self.verbosity >= 2:
                print(f"    Downloaded {len(files_to_download)} files")

            return local_dir

        except Exception as e:
            # Clean up directory on error (only if using temp dir)
            if self.use_temp_dir:
                shutil.rmtree(local_dir, ignore_errors=True)
            raise e

    def _cleanup(self):
        """Clean up temporary checkpoint directory."""
        if hasattr(self, "local_checkpoint_dir") and os.path.exists(
            self.local_checkpoint_dir
        ):
            shutil.rmtree(self.local_checkpoint_dir, ignore_errors=True)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a pandas DataFrame.

        Args:
            df: Input dataframe with text columns (must have all columns
               specified in task_config.text_columns)

        Returns:
            DataFrame with original columns + prediction columns:
            - Regression: pred_score (scaled), pred_score_scaled (original scale)
            - Binary classification: pred_logits, pred_score
            - Multi-class: pred_0_logits, pred_0_score, pred_1_logits, pred_1_score, ...

        Raises:
            ValueError: If required text columns are missing
        """
        # Validate input
        if len(df) == 0:
            if self.verbosity >= 1:
                print("[INFERENCE] Warning: Empty dataframe, returning as-is")
            return df

        # Check text columns exist
        missing_cols = [
            col for col in self.config.text_columns if col not in df.columns
        ]
        if len(missing_cols) > 0:
            raise ValueError(
                f"Missing required text columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        if self.verbosity >= 2:
            print(f"[INFERENCE] Running inference on {len(df):,} rows")
            print(f"  Text columns: {self.config.text_columns}")
            print(f"  Processing in batches of {self.batch_size}")

        # Process in batches
        all_predictions = []
        num_batches = (len(df) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(df), self.batch_size):
            batch_df = df.iloc[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            # Tokenize batch
            batch_inputs = self._tokenize_batch(batch_df)

            # Run inference
            with torch.no_grad():
                outputs = self.model(**batch_inputs)
                logits = outputs.logits

            # Post-process predictions
            batch_predictions = self._postprocess_predictions(logits)
            all_predictions.append(batch_predictions)

            if self.verbosity >= 2 and (
                batch_num % 10 == 0 or batch_num == num_batches
            ):
                print(f"  Processed batch {batch_num}/{num_batches}")

        # Concatenate all batch predictions
        final_predictions = {}
        for key in all_predictions[0].keys():
            # Concatenate arrays for this key across all batches
            final_predictions[key] = np.concatenate(
                [pred[key] for pred in all_predictions], axis=0
            )

        # Add predictions to dataframe
        result_df = df.copy()
        for key, values in final_predictions.items():
            result_df[key] = values

        if self.verbosity >= 2:
            print("[INFERENCE] Inference complete")
            print(f"  Added columns: {list(final_predictions.keys())}")

        return result_df

    def _tokenize_batch(self, batch_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of text data.

        Args:
            batch_df: Batch dataframe with text columns

        Returns:
            Dictionary with input_ids and attention_mask tensors on device
        """
        # Extract text columns in order
        texts = [batch_df[col].tolist() for col in self.config.text_columns]

        # Tokenize (sentence-pair format if multiple columns)
        tokenized = self.tokenizer(
            *texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )

        # Move to device
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        return tokenized

    def _postprocess_predictions(self, logits: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Post-process model logits into prediction columns.

        Applies calibration if calibrator is provided.

        Args:
            logits: Raw model output logits (batch_size, num_labels)

        Returns:
            Dictionary mapping column names to numpy arrays
        """
        predictions = {}

        if self.config.problem_type == ProblemType.REGRESSION:
            # Regression: single output value
            # logits shape: (batch_size, 1) or (batch_size,)
            preds = logits.squeeze().cpu().numpy()

            # Add uncalibrated predictions (scaled as model outputs)
            predictions["pred_score"] = preds

            # Apply calibration if calibrator provided
            if self.calibrator is not None:
                preds_calibrated = self.calibrator.predict(preds.reshape(-1, 1))
                if preds_calibrated.ndim > 1:
                    preds_calibrated = preds_calibrated.squeeze()
                predictions["pred_score_calibrated"] = preds_calibrated
                # Descale the calibrated predictions
                scale_factor = self.config.label_scale_factor
                predictions["pred_score_calibrated_scaled"] = preds_calibrated / scale_factor
            
            # Add descaled predictions (original scale)
            # Note: config is guaranteed to be RegressionTaskConfig here
            scale_factor = self.config.label_scale_factor
            predictions["pred_score_scaled"] = preds / scale_factor

        elif self.config.problem_type == ProblemType.CLASSIFICATION:
            if self.num_classes == 2:
                # Binary classification: single logit
                # logits shape: (batch_size, 1) or (batch_size,)
                logits_squeezed = logits.squeeze().cpu()

                # Add raw logits
                predictions["pred_logits"] = logits_squeezed.numpy()

                # Add sigmoid scores (uncalibrated)
                scores = torch.sigmoid(logits_squeezed).numpy()
                predictions["pred_score"] = scores

                # Apply calibration if calibrator provided
                if self.calibrator is not None:
                    scores_calibrated = self.calibrator.predict(scores)
                    if scores_calibrated.ndim > 1:
                        scores_calibrated = scores_calibrated.squeeze()
                    predictions["pred_score_calibrated"] = scores_calibrated

            else:
                # Multi-class classification: multiple logits
                # logits shape: (batch_size, num_classes)
                logits_cpu = logits.cpu()

                # Compute softmax scores (uncalibrated)
                scores = torch.softmax(logits_cpu, dim=-1).numpy()

                # Add per-class logits and uncalibrated scores
                for class_idx in range(self.num_classes):
                    predictions[f"pred_{class_idx}_logits"] = logits_cpu[
                        :, class_idx
                    ].numpy()
                    predictions[f"pred_{class_idx}_score"] = scores[:, class_idx]

                # Apply calibration if calibrators provided (list of calibrators)
                if self.calibrator is not None:
                    from fmcore.framework.bert_training.binary_evaluation import apply_multiclass_calibrator
                    
                    scores_calibrated = apply_multiclass_calibrator(
                        scores, self.calibrator, normalize=True
                    )
                    
                    # Add calibrated scores
                    for class_idx in range(self.num_classes):
                        predictions[f"pred_{class_idx}_score_calibrated"] = scores_calibrated[
                            :, class_idx
                        ]

        return predictions
