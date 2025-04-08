from abc import ABC
from typing import Dict

from bears.util import Registry
from fmcore.types.typed import MutableTyped


class BaseEvaluatorParams(MutableTyped, Registry, ABC):
    """
    Base class for evaluator parameters in Evaluators.

    Extends Registry to allow dynamic registration of custom evaluation parameters
    required for custom evaluators.
    """

    @classmethod
    def from_dict(cls, evaluator_type: str, evaluator_params: Dict) -> "BaseEvaluatorParams":
        """
        Creates an instance of the appropriate evaluator parameter subclass from a dictionary.

        This method is required by Pydantic validators to dynamically select parameter classes
        that are registered at runtime. This approach avoids the compile-time limitations of
        Pydantic discriminators.

        Args:
            evaluator_type (str): The type of evaluator to determine the correct subclass.
            evaluator_params (Dict): A dictionary of parameters for the evaluator.

        Returns:
            BaseEvaluatorParams: An instance of the resolved evaluator parameter subclass.
        """
        BaseEvaluatorParamsClass = BaseEvaluatorParams.get_subclass(key=evaluator_type)
        return BaseEvaluatorParamsClass(**evaluator_params)
