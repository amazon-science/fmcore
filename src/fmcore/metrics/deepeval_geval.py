from typing import Dict, Any
import logging
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase

from fmcore.adapters.deepeval_adapter import DeepEvalLLMAdapter
from fmcore.llm.base_llm import BaseLLM
from fmcore.metrics.base_metric import BaseMetric
from fmcore.types.enums.metric_enums import (
    MetricFramework,
    SupportedMetrics,
)
from fmcore.types.metric_types import (
    MetricConfig,
    MetricResult,
    TextGenerationMetricResult,
)
from fmcore.utils.introspection_utils import IntrospectionUtils
from fmcore.utils.deepeval_utils import DeepEvalUtils

logger = logging.getLogger(__name__)


class DeepEvalGEvalMetric(BaseMetric):
    aliases = [SupportedMetrics.DEEPEVAL_GEVAL]
    geval_metric_params: Dict[str, Any]

    @classmethod
    def _get_constructor_parameters(cls, *, metric_config: MetricConfig) -> Dict:
        llm = BaseLLM.of(llm_config=metric_config.llm_config)
        model = DeepEvalLLMAdapter(llm=llm)

        geval_metric_params = IntrospectionUtils.filter_params(
            func=GEval, params=metric_config.metric_params
        )
        if "evaluation_params" not in geval_metric_params:
            geval_metric_params["evaluation_params"] = DeepEvalUtils.infer_evaluation_params(
                field_mapping=metric_config.field_mapping
            )

        if not metric_config.framework:
            metric_config.framework = MetricFramework.DEEPEVAL

        geval_metric_params["model"] = model

        return {"config": metric_config, "geval_metric_params": geval_metric_params}

    def evaluate(self, data: Dict) -> MetricResult:
        """
        Evaluates the provided data using DeepEval's GEval metric.

        Args:
            data (Dict): Input data to evaluate

        Returns:
            MetricResult: The evaluation result containing the score and reason
        """
        metric = GEval(**self.geval_metric_params)
        # Construct LLMTestCase from the data using the utility
        test_case: LLMTestCase = DeepEvalUtils.map_data_to_testcase(
            data=data, field_mapping=self.config.field_mapping
        )

        metric.measure(test_case)

        # Create and return the result
        return TextGenerationMetricResult(score=metric.score, reason=metric.reason)

    async def aevaluate(self, data: Dict) -> MetricResult:
        """
        Asynchronously evaluates the provided data using DeepEval's GEval metric.

        Args:
            data (Dict): Input data to evaluate

        Returns:
            MetricResult: The evaluation result containing the score and reason
        """
        metric = GEval(**self.geval_metric_params)
        # Construct LLMTestCase from the data using the utility
        test_case: LLMTestCase = DeepEvalUtils.map_data_to_testcase(
            data=data, field_mapping=self.config.field_mapping
        )

        await metric.a_measure(test_case)

        # Create and return the result
        return TextGenerationMetricResult(score=metric.score, reason=metric.reason)
