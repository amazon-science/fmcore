from typing import *

import pandas as pd
from pydantic import constr, model_validator

from synthergent.cleaner.Cleaner import Cleaner
from synthergent.util import (
    as_list,
)


class StringCleaner(Cleaner):
    class Params(Cleaner.Params):
        cleaner: Callable
        col: Union[List[constr(min_length=1)], constr(min_length=1)]

        @model_validator(mode="before")
        @classmethod
        def _StringCleaner_check_params(cls, params: Dict) -> Dict:
            params["col"] = as_list(params["col"])
            return params

    def clean(
        self,
        data: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        for col in self.params.col:
            data[col] = data[col].apply(self.params.cleaner)
        return data
