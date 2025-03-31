from typing import Dict

from fmcore.mapper.base_mapper import BaseMapper
from asteval import Interpreter


class CriteriaCheckerMapper(BaseMapper[Dict, bool]):
    """
    A mapper that checks if input data meets specified criteria.
    """

    criteria: str

    def evaluate_expression(self, expression: str, context: dict):
        """
        Evaluates the criteria expression against the provided dictionary.

        AST interpreters are not inherently thread-safe, as they maintain an internal symbol table
        that is modified during execution. To ensure correctness, we instantiate a new Interpreter
        for each evaluation instead of sharing a global instance.

        Using a shared Interpreter would require synchronization mechanisms such as locks or
        thread-local storage to prevent concurrent modifications to the symbol table. However,
        benchmarking showed that even with optimizations, a shared, thread-safe implementation
        was at best only **30% faster** than creating a new instance per evaluation.

        Given that Interpreter instantiation is lightweight and avoids race conditions, the optimal
        approach is to create a new instance for each evaluation, populate its symbol table with
        the extracted values, and execute the criteria expression while maintaining correctness and performance.
        """
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
