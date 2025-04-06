from glom import glom
from typing import Dict

from fmcore.mapper.base_mapper import BaseMapper


class EqualityCheckerMapper(BaseMapper[Dict, bool]):
    """
    Mapper for evaluation that checks if the predicted label matches the ground truth.

    This mapper supports nested field access using dot-notation via `glom`.

    `glom` is used here to simplify retrieval of deeply nested fields from dictionaries without having
    to manually traverse each level. It allows keys like `"output.label"` or `"expected.labels.0"` to
    directly access nested values, making the mapper flexible and concise for complex data structures.

    Attributes:
        ground_truth_key (str): Dot-path to the ground truth value in the input dictionary.
        prediction_key (str): Dot-path to the predicted value in the input dictionary.
    """

    ground_truth_key: str
    prediction_key: str

    def map(self, data: Dict) -> bool:
        """
        Check if the predicted label matches the ground truth.

        Args:
            data (Dict): Input dictionary with potentially nested structure.

        Returns:
            bool: True if values at the specified keys match, False otherwise.
        """
        ground_truth = glom(data, self.ground_truth_key, default=None)
        prediction = glom(data, self.prediction_key, default=None)
        return ground_truth == prediction

    async def amap(self, data: Dict) -> bool:
        """
        Asynchronously check if the predicted label matches the ground truth.

        Args:
            data (Dict): Input dictionary with potentially nested structure.

        Returns:
            bool: True if values at the specified keys match, False otherwise.
        """
        return self.map(data)
