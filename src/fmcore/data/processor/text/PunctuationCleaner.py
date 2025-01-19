import string
from typing import *

from pydantic import constr

from fmcore.data.processor import SingleColumnProcessor, TextInputProcessor, TextOutputProcessor
from fmcore.util import String, is_null


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
            str.maketrans(string.punctuation, self.params.replacement_char * len(string.punctuation))
        )
