from typing import *

import pandas as pd

from synthergent.cleaner.Cleaner import Cleaner
from synthergent.util import (
    SampleSizeType,
)


class SubSample(Cleaner):
    class Params(Cleaner.Params):
        persist: bool = True
        size: SampleSizeType
        seed: int = 42

    def clean(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        return data
