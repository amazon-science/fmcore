from typing import Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage

from jinja2 import Template

from fmcore.mapper.base_mapper import BaseMapper


class TextPromptMapper(BaseMapper[Dict, BaseMessage]):
    """
    A mapper that transforms input data into text prompts using templates.
    """

    template: Template

    def map(self, data: Dict) -> BaseMessage:
        """
        Transform the input data into a message using the template.

        Args:
            data (Dict): Input data for template rendering

        Returns:
            BaseMessage: The generated message
        """
        rendered_text = self.template.render(**data)
        return HumanMessage(content=rendered_text)

    async def amap(self, data: Dict) -> BaseMessage:
        """
        Asynchronously transform the input data into a message.

        Args:
            data (Dict): Input data for template rendering

        Returns:
            BaseMessage: The generated message
        """
        return self.map(data)
