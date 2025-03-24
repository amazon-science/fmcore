from typing import Any, Dict, Optional, List

import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
from fmcore.prompt_tuner.dspy.datasets.base_dataset import DspyDataset
from fmcore.prompt_tuner.dspy.optimizers.base_dspy_optimizer import (
    BaseDspyOptimizer,
)
from fmcore.prompt_tuner.dspy.utils.commons import DSPyUtils
from fmcore.types.enums.prompt_tuner_enums import DspyOptimizerType
from fmcore.types.prompt_tuner_types import TunedPrompt, PromptTunerResult
from fmcore.utils.introspection_utils import IntrospectionUtils


class MIPROV2Optimizer(BaseDspyOptimizer):
    """
    Optimizer class that uses DSPy's MIPROv2 for prompt optimization.

    This class supports providing default parameters while allowing runtime
    parameter overrides through a strongly-typed configuration class.
    """

    aliases = [DspyOptimizerType.MIPRO_V2]

    def optimize(
        self, dataset: DspyDataset, optimizer_params: Optional[Dict[str, Any]] = None
    ) -> List[dspy.Module]:
        """
        Optimize prompts using MIPROv2 with default parameters and runtime overrides.

        This method:
        1. Initializes MIPROv2 optimizer with filtered parameters
        2. Compiles and runs optimization on training data
        3. Extracts prompts from candidate programs and scores them
        4. Returns sorted results with best prompts first

        Args:
            dataset: DspyDataset containing training and validation data
            optimizer_params: Optional dictionary of parameters to override defaults

        Returns:
            PromptTunerResult containing sorted list of optimized prompts and their scores
        """
        # Initialize MIPROv2 optimizer with filtered constructor params
        constructor_params = IntrospectionUtils.filter_params(
            func=MIPROv2, params=optimizer_params or {}
        )
        optimizer = MIPROv2(
            metric=self.evaluate,
            prompt_model=self.teacher,
            task_model=self.student,
            **constructor_params,
        )

        # Run optimization with filtered compile params
        compile_params = IntrospectionUtils.filter_params(
            func=MIPROv2.compile, params=optimizer_params or {}
        )
        optimized_program = optimizer.compile(
            student=self.module,
            trainset=dataset.train,
            valset=dataset.dev,
            requires_permission_to_run=False,
            **compile_params,
        )

        optimized_candidates = [candidate["program"].predictor.predict
                             for candidate in optimized_program.candidate_programs]
        return optimized_candidates
