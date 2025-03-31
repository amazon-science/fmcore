from typing import Dict

from fmcore.mapper.base_mapper import BaseMapper
from asteval import Interpreter


class CriteriaCheckerMapper(BaseMapper[Dict, bool]):
    """
    A mapper that checks if input data meets specified criteria.
    """

    criteria: str

    def evaluate_expression(self, expression: str, context: dict):
        aeval = Interpreter()
        aeval.symtable.update(context)  # Load dictionary values
        return aeval(expression)

    def map(self, data: Dict) -> bool:
        """
        Check if the input data meets the specified criteria.

        Args:
            data (Dict): The input data to check

        Returns:
            bool: True if criteria is met, False otherwise
        """
        return self.evaluate_expression(self.criteria, data)

    async def amap(self, data: Dict) -> bool:
        """
        Asynchronously check if the input data meets the specified criteria.

        Args:
            data (Dict): The input data to check

        Returns:
            bool: True if criteria is met, False otherwise
        """
        return self.map(data)
