import dspy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, List

from fmcore.prompt_tuner.dspy.datasets.base_dataset import DspyDataset
from fmcore.types.enums.prompt_tuner_enums import DspyOptimizerType
from fmcore.types.prompt_tuner_types import PromptTunerResult
from fmcore.types.typed import MutableTyped
from bears.util import Registry


class BaseDspyOptimizer(MutableTyped, Registry, ABC):
    student: dspy.LM
    teacher: Optional[dspy.LM]
    module: dspy.Module
    evaluate: Callable

    @classmethod
    def of(
        cls,
        optimizerType: DspyOptimizerType,
        student: dspy.LM,
        teacher: Optional[dspy.LM],
        module: dspy.Module,
        evaluate: Callable,
        **kwargs,
    ) -> str:
        BaseDspyOptimizerClass = BaseDspyOptimizer.get_subclass(key=optimizerType)
        return BaseDspyOptimizerClass(
            student=student, teacher=teacher, module=module, evaluate=evaluate
        )

    @abstractmethod
    def optimize(self, dataset: DspyDataset, optimzer_params: Dict[str, Any]) -> List[dspy.Module]:
        pass
