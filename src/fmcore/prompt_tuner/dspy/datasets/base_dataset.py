from typing import ClassVar, Dict
import pandas as pd
from dspy.datasets.dataset import Dataset
from dspy.datasets import DataLoader

from fmcore.prompt_tuner.types.prompt_tuner_types import PromptConfig
from fmcore.types.enums.dataset_enums import DatasetType


class DspyDataset(Dataset):
    """
    A dataset class for DSPy framework to handle training, validation, and test data.

    This class transforms input data (from pandas DataFrame) into DSPy examples for training,
    validation, and testing. The `input_keys` hold the names of the fields that are required
    for input in the prompts.

    Attributes:
        loader (ClassVar[DataLoader]): The DSPy DataLoader used to load data from DataFrame.
        _train: Processed training data.
        _dev: Processed validation data.
        _test: Processed test data.
        input_keys: List of input field names based on the prompt configuration.
    """

    loader: ClassVar[DataLoader] = DataLoader()

    def __init__(
        self, data: Dict[DatasetType, pd.DataFrame], prompt_config: PromptConfig, *args, **kwargs
    ) -> None:
        """
        Initializes the DspyDataset instance.

        Args:
            data: A dictionary with dataset types as keys (TRAIN, VAL, TEST) and corresponding pandas DataFrames.
            prompt_config: The configuration containing input fields that define which columns are used in the prompt.
            *args: Additional positional arguments passed to the parent class constructor.
            **kwargs: Additional keyword arguments passed to the parent class constructor.

        Initializes:
            - Converts pandas DataFrames to DSPy examples using the DataLoader.
            - Stores the input field names as `input_keys`.
        """
        super().__init__(*args, **kwargs)

        # Extract input field names from the prompt configuration
        input_keys = [field.name for field in prompt_config.input_fields]

        # Get the list of columns from the training data to ensure the right fields are loaded
        fields = data[DatasetType.TRAIN].columns.tolist()

        # Load the data into DSPy examples for train, validation, and test datasets
        train_examples = self.loader.from_pandas(df=data[DatasetType.TRAIN], fields=fields)
        dev_examples = self.loader.from_pandas(df=data[DatasetType.VAL], fields=fields)
        test_examples = self.loader.from_pandas(df=data[DatasetType.TEST], fields=fields)

        # Store the processed datasets
        self._train = train_examples
        self._dev = dev_examples
        self._test = test_examples
        self.input_keys = input_keys
