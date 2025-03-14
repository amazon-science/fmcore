from typing import ClassVar
import pandas as pd
from dspy.datasets.dataset import Dataset
from dspy.datasets import DataLoader
from sklearn.model_selection import train_test_split

from fmcore.experimental.types.prompt_tuner_types import PromptConfig


class DspyDataset(Dataset):
    loader: ClassVar[DataLoader] = DataLoader()

    def __init__(self, data: pd.DataFrame, prompt_config: PromptConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        input_keys = [field.name for field in prompt_config.input_fields]
        examples = self.loader.from_pandas(df=data, fields=data.columns.tolist())

        train_examples, test_examples = train_test_split(
            examples, train_size=0.6, random_state=self.train_seed
        )
        dev_examples, test_examples = train_test_split(
            test_examples, train_size=0.5, random_state=self.dev_seed
        )

        self._train = train_examples
        self._dev = dev_examples
        self._test = test_examples
        self.input_keys = input_keys
