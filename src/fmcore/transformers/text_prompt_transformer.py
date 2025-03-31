from jinja2 import Template
from langchain.schema import BaseMessage
from typing import Dict

from langchain_core.messages import HumanMessage

from fmcore.transformers.base_transformer import BaseTransformer, I, O


class TextPromptTransformer(BaseTransformer[Dict, BaseMessage]):
    """
    A transformer that initializes a Jinja template with a given prompt string
    and renders it into a BaseMessage.
    """

    template: Template

    def __init__(self, prompt_template: str):
        """
        Initializes the transformer with a Jinja template.
        """
        # Parsing a Jinja template is an expensive operation. We benchmarked two approaches:
        # 1. Creating a new template for each render before rendering (100k renders took 284.8903 sec).
        # 2. Using a single pre-compiled template and rendering it multiple times (100k renders took 3.4328 sec).
        # The second approach resulted in an 85x performance improvement.

        # Since the template remains unchanged across multiple evaluators, we create it once
        # and reuse it throughout the evaluator to optimize performance.
        template = Template(prompt_template, autoescape=True)

        super().__init__(template=template)

    def transform(self, data: Dict) -> BaseMessage:
        """
        Renders the Jinja template with the provided data and converts it into a BaseMessage.
        """
        rendered_prompt = self.template.render(data)
        return HumanMessage(content=rendered_prompt)

    async def atransform(self, data: I) -> O:
        return self.transform(data=data)
