from typing import Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from bears import FileMetadata, ScalableDataFrame
from bears.reader import Reader
from fmcore.types.enums.dataset_enums import DatasetType


class DatasetUtils:
    """Utility class for dataset operations like loading, splitting, and preprocessing."""

    # Standard split ratios
    TRAIN_RATIO = 0.6  # When splitting train into three parts (train/val/test)
    VAL_RATIO_FROM_REMAINING = 0.5  # For splitting the remainder into val and test (50/50)
    TRAIN_RATIO_WHEN_TEST_EXISTS = 0.75  # When splitting train to get validation (75/25)

    @staticmethod
    def load_and_split_datasets(
        inputs: Dict[DatasetType, FileMetadata],
    ) -> Dict[DatasetType, pd.DataFrame]:
        """
        Load datasets and create splits as needed, ensuring all three splits (train, val, test)
        are always present in the returned dictionary.

        Handles several scenarios:
        1. Use provided train/val/test datasets if all are available
        2. Split train 60/20/20 if only train is provided
        3. Split train 75/25 to create validation set if train and test are provided

        Args:
            inputs: Dictionary mapping DatasetType to FileMetadata

        Returns:
            Dictionary mapping DatasetType to corresponding pandas DataFrame with
            guaranteed keys for DatasetType.TRAIN, DatasetType.VAL, and DatasetType.TEST
        """
        data: Dict[DatasetType, pd.DataFrame] = {}

        # Load available datasets
        for dataset_type in [DatasetType.TRAIN, DatasetType.VAL, DatasetType.TEST]:
            if dataset_type in inputs:
                data[dataset_type] = DatasetUtils.read_dataset(file_metadata=inputs[dataset_type])

        # No train data available - nothing to split
        if DatasetType.TRAIN not in data:
            raise ValueError("Training data must be provided")

        # Case 1: Only train data is available, create 60/20/20 split
        if DatasetType.VAL not in data and DatasetType.TEST not in data:
            # First split train into train and temp (60/40)
            train_df, remaining = train_test_split(
                data[DatasetType.TRAIN], train_size=DatasetUtils.TRAIN_RATIO, random_state=42
            )
            # Split temp into val and test (50/50 of the 40%, which gives 20/20)
            val_df, test_df = train_test_split(
                remaining, train_size=DatasetUtils.VAL_RATIO_FROM_REMAINING, random_state=42
            )

            data[DatasetType.TRAIN] = train_df.reset_index(drop=True)
            data[DatasetType.VAL] = val_df.reset_index(drop=True)
            data[DatasetType.TEST] = test_df.reset_index(drop=True)

        # Case 2: Train and test data available but no validation - create validation from train (75/25)
        elif DatasetType.TEST in data and DatasetType.VAL not in data:
            train_df, val_df = train_test_split(
                data[DatasetType.TRAIN],
                train_size=DatasetUtils.TRAIN_RATIO_WHEN_TEST_EXISTS,
                random_state=42,
            )

            data[DatasetType.TRAIN] = train_df.reset_index(drop=True)
            data[DatasetType.VAL] = val_df.reset_index(drop=True)

        # Case 3: Train and val data available but no test - create test from val (50/50)
        elif DatasetType.VAL in data and DatasetType.TEST not in data:
            val_df, test_df = train_test_split(data[DatasetType.VAL], train_size=0.5, random_state=42)

            data[DatasetType.VAL] = val_df.reset_index(drop=True)
            data[DatasetType.TEST] = test_df.reset_index(drop=True)

        # Verify all dataset types are present
        for dataset_type in [DatasetType.TRAIN, DatasetType.VAL, DatasetType.TEST]:
            if dataset_type not in data:
                raise ValueError(f"Failed to generate {dataset_type} dataset")

        return data

    @staticmethod
    def read_dataset(file_metadata: FileMetadata) -> pd.DataFrame:
        """
        Read a dataset from the provided file metadata.

        Args:
            file_metadata: Metadata for the file to read

        Returns:
            DataFrame containing the dataset
        """
        reader: Reader = Reader.of(file_format=file_metadata.format)
        sdf: ScalableDataFrame = reader.read(source=file_metadata)
        return sdf.as_pandas()
