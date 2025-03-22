import json_repair
from typing import Dict, List
from langchain_core.messages import BaseMessage, HumanMessage
from jinja2 import Template
from fmcore.llm.base_llm import BaseLLM
from fmcore.metrics.base_metric import BaseMetric
from fmcore.types.enums.metric_enums import SupportedMetrics
from fmcore.types.metric_types import (
    CustomMetricResult,
    MetricConfig,
    MetricResult,
)


class CustomMetric(BaseMetric):
    aliases = [SupportedMetrics.CUSTOM]

    metric_name: str
    llm: BaseLLM
    prompt_template: Template

    @classmethod
    def _get_constructor_parameters(cls, *, metric_config: MetricConfig) -> Dict:
        llm: BaseLLM = BaseLLM.of(llm_config=metric_config.llm_config)

        prompt: str = metric_config.metric_params["prompt"]
        prompt_template = Template(prompt)

        metric_name: str = metric_config.metric_params["name"]

        return {
            "config": metric_config,
            "prompt_template": prompt_template,
            "llm": llm,
            "metric_name": metric_name,
        }

    def evaluate(self, data: Dict) -> MetricResult:
        prompt: str = self.prompt_template.render(**data)
        messages: List[BaseMessage] = [HumanMessage(content=prompt)]
        response: BaseMessage = self.llm.invoke(messages=messages)
        result: Dict = json_repair.loads(response.content)

        return CustomMetricResult(**result)

    async def aevaluate(self, data: Dict) -> MetricResult:
        prompt: str = self.prompt_template.render(**data)
        messages: List[BaseMessage] = [HumanMessage(content=prompt)]
        response: BaseMessage = await self.llm.ainvoke(messages=messages)
        result: Dict = json_repair.loads(response.content)

        return CustomMetricResult(**result)
