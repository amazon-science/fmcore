from typing import *
from fmcore.data.processor import SingleColumnProcessor, TextInputProcessor, TextOutputProcessor
from fmcore.util import AutoEnum, auto, String, is_null
import pandas as pd
import string
from pydantic import constr


class PunctuationCleaner(SingleColumnProcessor, TextInputProcessor, TextOutputProcessor):
    """
    Replaces punctuations with spaces.
    """

    class Params(SingleColumnProcessor.Params):
        replacement_char: constr(min_length=1) = String.SPACE

    def transform_single(self, data: Optional[str]) -> Optional[str]:
        if is_null(data):
            return None
        return data.translate(
            str.maketrans(
                string.punctuation,
                self.params.replacement_char * len(string.punctuation)
            )
        )
