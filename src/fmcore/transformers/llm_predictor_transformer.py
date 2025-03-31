from abc import ABC, abstractmethod
from typing import List, Any
from langchain.schema import BaseMessage

from fmcore.llm import BaseLLM
from fmcore.llm.types.llm_types import LLMConfig
from fmcore.transformers.base_transformer import BaseTransformer


class LLMPredictor(BaseTransformer[List[BaseMessage], BaseMessage], ABC):
    """
    A concrete LLM predictor that initializes an LLM configuration using Pydantic and uses it
    to process a list of BaseMessage objects, generating a single response message.
    """

    llm: BaseLLM

    def __init__(self, llm_config: LLMConfig):
        llm = BaseLLM.of(llm_config=llm_config)
        super().__init__(llm=llm)

    def transform(self, data: List[BaseMessage]) -> BaseMessage:
        """
        Synchronously processes the input messages and returns the LLM prediction.

        Args:
            data (List[BaseMessage]): A list of messages to be processed.

        Returns:
            BaseMessage: The generated response message.
        """
        return self.llm.invoke(messages=data)

    async def atransform(self, data: List[BaseMessage]) -> BaseMessage:
        """
        Asynchronously processes the input messages and returns the LLM prediction.

        Args:
            data (List[BaseMessage]): A list of messages to be processed.

        Returns:
            BaseMessage: The generated response message.
        """
        return await self.llm.ainvoke(messages=data)
