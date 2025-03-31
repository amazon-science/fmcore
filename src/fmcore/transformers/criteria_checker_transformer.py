from typing import Dict

from asteval import Interpreter

from fmcore.transformers.base_transformer import BaseTransformer, I, O


class CriteriaChecker(BaseTransformer[Dict, bool]):
    """
    A transformer that evaluates a given criteria condition on an input dictionary.
    """

    def __init__(self, criteria: str):
        """
        Initializes the evaluator with a criteria condition.
        The criteria should be a Python expression where dictionary keys can be referenced directly.
        """
        super().__init__(criteria=criteria)

    def transform(self, data: Dict) -> bool:
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

        expression_evaluator = Interpreter()
        expression_evaluator.symtable.update(data)
        return expression_evaluator(self.criteria)

    async def atransform(self, data: Dict) -> bool:
        return self.transform(data=data)
