from typing import List
from langchain.schema import BaseMessage

from fmcore.llm import BaseLLM
from fmcore.mapper.base_mapper import BaseMapper
from fmcore.utils.async_utils import AsyncUtils


class LLMInferenceMapper(BaseMapper[List[BaseMessage], BaseMessage]):
    """
    A concrete LLM inference mapper that initializes an LLM configuration using Pydantic and uses it
    to process a list of BaseMessage objects, generating a single response message.
    """

    llm: BaseLLM

    def map(self, data: List[BaseMessage]) -> BaseMessage:
        """
        Synchronously processes the input messages and returns the LLM prediction.
        Args:
            data (List[BaseMessage]): A list of messages to be processed.
        Returns:
            BaseMessage: The generated response message.
        """
        return AsyncUtils.execute(self.llm.ainvoke(messages=data))

    async def amap(self, data: List[BaseMessage]) -> BaseMessage:
        """
        Asynchronously processes the input messages and returns the LLM prediction.
        Args:
            data (List[BaseMessage]): A list of messages to be processed.
        Returns:
            BaseMessage: The generated response message.
        """
        return await self.llm.ainvoke(messages=data)
