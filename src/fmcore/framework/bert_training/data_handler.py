"""
Data handler for loading and preprocessing datasets.

Handles:
- Loading datasets from S3 parquet files
- Auto-detecting number of classes for classification tasks
- Tokenizing text columns in sentence-pair format
- Label validation and scaling
"""

from typing import Any, Dict, Optional, Tuple

import polars as pl
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from fmcore.framework.bert_training.config import ProblemType, RegressionTaskConfig, TaskConfig


class DataHandler:
    """Handler for dataset loading, preprocessing, and tokenization."""

    def __init__(self, config: TaskConfig, tokenizer: AutoTokenizer, is_rank_0: bool):
        """
        Initialize data handler.

        Args:
            config: Task configuration
            tokenizer: Initialized tokenizer
            is_rank_0: Whether this is the rank 0 worker (for logging)
        """
        self.config = config
        self.tokenizer = tokenizer
        self.is_rank_0 = is_rank_0
        self.num_classes: Optional[int] = None  # For classification only

    def load_and_prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Load train/val datasets from S3 and prepare them for training.

        For classification tasks, auto-detects num_classes from the data.

        Returns:
            Tuple of (train_dataset, val_dataset) as HuggingFace Dataset objects

        Raises:
            ValueError: If datasets are empty, columns are missing, or
                       labels are not properly mapped
        """
        if self.is_rank_0:
            print("[DATA] Loading datasets from S3...")
            print(f"  Train: {self.config.train_data_path}")
            print(f"  Val: {self.config.val_data_path}")

        # Load datasets using HuggingFace datasets library
        train_dataset = load_dataset(
            "parquet", data_files=self.config.train_data_path, split="train"
        )

        val_dataset = load_dataset(
            "parquet",
            data_files=self.config.val_data_path,
            split="train",  # Use "train" split even for val (we specify our own files)
        )

        # Validate datasets
        self._validate_dataset(train_dataset, "train")
        self._validate_dataset(val_dataset, "validation")

        if self.is_rank_0:
            print(f"  Train samples: {len(train_dataset):,}")
            print(f"  Val samples: {len(val_dataset):,}")

        # For classification, auto-detect number of classes
        if self.config.problem_type == ProblemType.CLASSIFICATION:
            self.num_classes = self._auto_detect_num_classes(
                self.config.train_data_path
            )
            if self.is_rank_0:
                print(f"  Auto-detected {self.num_classes} classes")

        # Tokenize datasets
        if self.is_rank_0:
            print("[DATA] Tokenizing datasets...")

        train_dataset = train_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            batch_size=100_000,  # Large batch size for efficiency
            desc="Tokenizing train dataset",
        )

        val_dataset = val_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            batch_size=100_000,
            desc="Tokenizing validation dataset",
        )

        if self.is_rank_0:
            print(f"  Tokenized {len(train_dataset):,} train samples")
            print(f"  Tokenized {len(val_dataset):,} val samples")

        return train_dataset, val_dataset

    def _validate_dataset(self, dataset: Dataset, split_name: str):
        """
        Validate that dataset has required columns and is not empty.

        Args:
            dataset: Dataset to validate
            split_name: Name of the split (for error messages)

        Raises:
            ValueError: If validation fails
        """
        if len(dataset) == 0:
            raise ValueError(f"{split_name} dataset is empty")

        # Check text columns exist
        for col in self.config.text_columns:
            if col not in dataset.column_names:
                raise ValueError(
                    f"Text column '{col}' not found in {split_name} dataset. "
                    f"Available columns: {dataset.column_names}"
                )

        # Check label column exists
        if self.config.label_column not in dataset.column_names:
            raise ValueError(
                f"Label column '{self.config.label_column}' not found in {split_name} dataset. "
                f"Available columns: {dataset.column_names}"
            )

    def _auto_detect_num_classes(self, data_path: str) -> int:
        """
        Use polars to efficiently count unique labels in classification task.

        Validates that labels are pre-mapped to 0, 1, 2, ..., n-1.

        Args:
            data_path: S3 path to parquet file

        Returns:
            Number of unique classes

        Raises:
            ValueError: If labels are not properly mapped to consecutive integers
        """
        # Use polars for efficient unique value detection
        df = pl.read_parquet(data_path)
        unique_labels = df[self.config.label_column].unique().sort()

        # Convert to list for validation
        unique_labels_list = unique_labels.to_list()

        # Validate labels are 0, 1, 2, ..., n-1
        expected = list(range(len(unique_labels_list)))
        if unique_labels_list != expected:
            raise ValueError(
                f"Labels must be pre-mapped to 0, 1, 2, ..., n-1. "
                f"Found: {unique_labels_list}. "
                f"Expected: {expected}"
            )

        num_classes = len(unique_labels_list)

        if self.is_rank_0:
            print(f"  Label validation passed: {unique_labels_list}")

        return num_classes

    def _tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize multiple text columns in sentence-pair format.

        BERT tokenizer automatically handles sentence pairs:
        tokenizer(text1, text2) -> [CLS] text1 [SEP] text2 [SEP]

        Args:
            examples: Batch of examples from HuggingFace Dataset

        Returns:
            Dictionary with tokenized inputs and labels
        """
        # Extract text columns
        # Pass all text columns as separate arguments to tokenizer
        # E.g., tokenizer(query, product) -> [CLS] query [SEP] product [SEP]
        texts = [examples[col] for col in self.config.text_columns]

        # Tokenize
        tokenized = self.tokenizer(
            *texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
        )

        # Add labels
        labels = examples[self.config.label_column]

        # For regression, optionally apply scaling
        if isinstance(self.config, RegressionTaskConfig):
            if self.config.label_scale_factor != 1.0:
                labels = [
                    float(label) * self.config.label_scale_factor for label in labels
                ]

        tokenized["labels"] = labels

        return tokenized
