from typing import List, TypeVar
import random

T = TypeVar('T')


class CollectionUtils:
    @staticmethod
    def split_into_equal_parts(items: List[T], num_parts: int, randomize: bool = False) -> List[List[T]]:
        """
        Splits a list into roughly equal-sized parts.

        If randomize is True, elements are shuffled before splitting.
        Otherwise, elements are kept in original order.

        If the list can't be evenly divided, the first few parts
        will have one more element than the rest.

        Args:
            items (List): The list to split.
            num_parts (int): The number of parts to split into.
            randomize (bool): Whether to shuffle elements before splitting.

        Returns:
            List[List]: A list of sublists.

        Raises:
            ValueError: If input is invalid.
        """
        if not isinstance(items, list):
            raise ValueError("items must be a list")
        if not isinstance(num_parts, int) or num_parts <= 0:
            raise ValueError("num_parts must be a positive integer")
        if not items:
            raise ValueError("items list cannot be empty")

        working_items = items[:]
        if randomize:
            random.shuffle(working_items)

        base_size = len(working_items) // num_parts
        extras = len(working_items) % num_parts

        result = []
        index = 0
        for i in range(num_parts):
            chunk_size = base_size + (1 if i < extras else 0)
            result.append(working_items[index: index + chunk_size])
            index += chunk_size

        return result
