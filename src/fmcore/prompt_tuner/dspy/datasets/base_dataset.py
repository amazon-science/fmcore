from typing import ClassVar, Dict
import pandas as pd
from dspy.datasets.dataset import Dataset
from dspy.datasets import DataLoader

from fmcore.types.prompt_tuner_types import PromptConfig

from fmcore.types.enums.dataset_enums import DatasetType


class DspyDataset(Dataset):
    loader: ClassVar[DataLoader] = DataLoader()

    def __init__(
        self, data: Dict[DatasetType, pd.DataFrame], prompt_config: PromptConfig, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        input_keys = [field.name for field in prompt_config.input_fields]
        fields = data[DatasetType.TRAIN].columns.tolist()

        train_examples = self.loader.from_pandas(df=data[DatasetType.TRAIN], fields=fields)
        dev_examples = self.loader.from_pandas(df=data[DatasetType.VAL], fields=fields)
        test_examples = self.loader.from_pandas(df=data[DatasetType.TEST], fields=fields)

        self._train = train_examples
        self._dev = dev_examples
        self._test = test_examples
        self.input_keys = input_keys
