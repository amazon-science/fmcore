import dspy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from dspy.teleprompt import Teleprompter
from dspy import Example

from fmcore.experimental.prompt_tuner.dspy.datasets.base_dataset import DspyDataset
from fmcore.experimental.types.enums.prompt_tuner_enums import DspyOptimizerType
from fmcore.experimental.types.prompt_tuner_types import PromptTunerResult
from fmcore.experimental.types.typed import MutableTyped
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
    def optimize(self, dataset: DspyDataset, optimzer_params: Dict[str, Any]) -> PromptTunerResult:
        pass
