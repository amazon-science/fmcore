import random
from typing import List, TypeVar

T = TypeVar("T")


class CollectionUtils:
    @staticmethod
    def split_into_equal_parts_randomized(items: List[T], num_parts: int) -> List[List[T]]:
        """
        Randomly splits a list into roughly equal-sized parts.

        Elements are shuffled before being split. If the list can't be evenly divided,
        the first few parts will have one more element than the rest.

        Args:
            items (List): The list to split.
            num_parts (int): The number of parts to split into.

        Returns:
            List[List]: A list of sublists.

        Raises:
            ValueError: If input is invalid.

        Example:
            >>> CollectionUtils.split_randomly([1, 2, 3, 4, 5], 2)
            [[3, 1, 4], [2, 5]]
        """
        if not isinstance(items, list):
            raise ValueError("items must be a list")
        if not isinstance(num_parts, int) or num_parts <= 0:
            raise ValueError("num_parts must be a positive integer")
        if not items:
            raise ValueError("items list cannot be empty")

        random.shuffle(items)
        base_size = len(items) // num_parts
        extras = len(items) % num_parts

        result = []
        index = 0
        for i in range(num_parts):
            chunk_size = base_size + (1 if i < extras else 0)
            result.append(items[index : index + chunk_size])
            index += chunk_size

        return result
