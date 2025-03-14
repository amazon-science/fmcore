import dspy
import mlflow.dspy
from pandas import DataFrame
from typing import Callable, Dict, Optional, List, Tuple, Type, Any, Union

from dspy.teleprompt import Teleprompter
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.teleprompt.bootstrap import BootstrapFewShot
from dspy import Signature, Module

from fmcore.experimental.metrics.base_metric import BaseMetric
from fmcore.experimental.prompt_tuner.base_prompt_tuner import BasePromptTuner
from fmcore.experimental.types.enums.prompt_tuner_enums import PromptTunerFramework
from fmcore.experimental.types.prompt_tuner_types import PromptTunerConfig

from fmcore.experimental.adapters.dspy_adapter import DSPyLLMAdapter
from fmcore.experimental.utils.introspection_utils import IntrospectionUtils
from fmcore.experimental.prompt_tuner.utils.dspy_utils import DSPyUtils
from asteval import Interpreter


class DSPyPromptTuner(BasePromptTuner):
    """
    A prompt tuner implementation using the DSPy framework.
    
    This class provides functionality to optimize prompts using various DSPy optimizers
    such as MIPROv2 or BootstrapFewShot. It uses a student model for generating responses
    and evaluates them using a configured metric to iteratively improve the prompt.
    
    Attributes:
        aliases (List[PromptTunerFramework]): Framework identifiers for this tuner.
        student (dspy.LM): The student language model used for prompt optimization.
        teacher (Optional[dspy.LM]): The teacher language model used in some optimization techniques.
        optimizer_metric (BaseMetric): The metric used to evaluate prompt performance.
    """
    
    aliases = [PromptTunerFramework.DSPY]
    student: dspy.LM
    teacher: Optional[dspy.LM]
    optimizer_metric: BaseMetric

    @classmethod
    def _get_constructor_parameters(cls, *, config: PromptTunerConfig) -> Dict[str, Any]:
        """
        Creates and configures the necessary components for DSPy prompt tuning.
        
        Args:
            config: Configuration containing all necessary parameters for the prompt tuner.
                   Must include student model config and optionally teacher model config.
        
        Returns:
            Dictionary of parameters needed to initialize the DSPyPromptTuner instance.
        """
        # Initialize student model and configure DSPy to use it
        student_model = DSPyLLMAdapter(llm_config=config.optimzer_config.student_config)
        dspy.configure(lm=student_model)

        # Initialize teacher model (or use student if not specified)
        if config.optimzer_config.teacher_config:
            teacher_model = DSPyLLMAdapter(llm_config=config.optimzer_config.teacher_config)
        else:
            teacher_model = student_model
            
        # Initialize metric for optimization
        optimizer_metric = BaseMetric.of(metric_config=config.optimzer_config.metric_config)

        return {
            "student": student_model,
            "teacher": teacher_model,
            "optimizer_metric": optimizer_metric,
            "config": config,
        }

    def _create_evaluation_function(self) -> Callable:
        """
        Creates an evaluation function that uses the configured metric.
        
        The function evaluates DSPy predictions by applying the metric and interpreting
        the criteria expression to determine the quality of the prediction.
        
        Returns:
            A callable function that takes an example and prediction and returns a 
            numerical or boolean evaluation score.
        """
        # Store criteria once to avoid re-fetching it in each evaluation call
        criteria = self.optimizer_metric.config.metric_params["criteria"]

        def evaluate_func(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> Union[float, bool]:
            """
            Evaluates a single example-prediction pair using the configured metric.
            
            Args:
                example: The DSPy example containing input data
                prediction: The model's prediction to evaluate
                trace: Optional trace information from DSPy (not used)
                
            Returns:
                Evaluation score as determined by the configured criteria
            """
            # Get evaluation results from the metric
            evaluation_response: dict = DSPyUtils.evaluate(
                example=example, 
                prediction=prediction, 
                metric=self.optimizer_metric
            )

            # Create a new Interpreter instance for each call to ensure thread safety
            expression_evaluator = Interpreter()
            expression_evaluator.symtable.update(evaluation_response)
            return expression_evaluator(criteria)

        return evaluate_func

    def tune(self, data: DataFrame) -> str:
        """
        Tunes a prompt using the configured DSPy optimizer and training data.
        
        This method:
        1. Converts the input data to DSPy examples
        2. Creates a DSPy signature and module based on the prompt configuration
        3. Configures an evaluation function using the specified metric
        4. Applies the DSPy optimizer to generate an optimized prompt
        
        Args:
            data: DataFrame containing the training data with input and expected output fields
        
        Returns:
            The optimized prompt as a string
        
        Raises:
            ValueError: If the optimization process fails or returns invalid results
        """

        import mlflow
        mlflow.dspy.autolog(log_traces=True, log_traces_from_compile=True, log_traces_from_eval=True, disable=False, silent=False)

        # Convert data to DSPy examples
        dspy_examples = DSPyUtils.convert_to_dspy_examples(
            data=data, 
            prompt_config=self.config.prompt_config
        )

        # Create signature and module separately
        signature: Signature = DSPyUtils.create_dspy_signature(
            prompt_config=self.config.prompt_config
        )
        module: Module = DSPyUtils.create_dspy_module(signature=signature)

        # Create evaluation function
        evaluate_func = self._create_evaluation_function()

        # Create optimizer
        optimizer: Teleprompter = DSPyUtils.get_optimizer(
            student=self.student,
            teacher=self.teacher,
            optimzer_config=self.config.optimzer_config,
            evaluate_func=evaluate_func,
        )

        # Filter optimizer parameters to only include those accepted by the compile method
        filtered_optimizer_params = IntrospectionUtils.filter_params(
            func=optimizer.compile, 
            params=self.config.optimzer_config.params
        )
        
        # Compile the module with the optimizer
        optimized_module = optimizer.compile(
            student=module,
            trainset=dspy_examples,
            requires_permission_to_run=False,
            **filtered_optimizer_params,
        )

        dspy.inspect_history(optimized_module)
        
        optimized_module.signature.prompt
