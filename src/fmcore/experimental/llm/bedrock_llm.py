import asyncio
import random
from typing import List, Iterator

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, BaseMessageChunk

from fmcore.experimental.factory.bedrock_factory import (
    BedrockFactory,
    BedrockClientProxy,
)
from fmcore.experimental.llm.base_llm import BaseLLM
from fmcore.experimental.types.enums.provider_enums import ProviderType
from fmcore.experimental.types.llm_types import LLMConfig


class BedrockLLM(BaseLLM, BaseModel):
    aliases = [ProviderType.BEDROCK]

    bedrock_clients: List[BedrockClientProxy] = Field(default_factory=list)

    @classmethod
    def _get_constructor_parameters(cls, *, llm_config: LLMConfig) -> dict:
        bedrock_clients = BedrockFactory.create_bedrock_clients(llm_config=llm_config)
        return {"config": llm_config, "bedrock_clients": bedrock_clients}

    def get_random_client(self) -> BedrockClientProxy:
        if not self.bedrock_clients:
            raise ValueError("No Bedrock clients available.")
        return random.choice(self.bedrock_clients)

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        bedrock_proxy: BedrockClientProxy = self.get_random_client()
        return bedrock_proxy.client.invoke(input=messages)

    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        bedrock_proxy: BedrockClientProxy = self.get_random_client()
        async with bedrock_proxy.rate_limiter:
            return await bedrock_proxy.client.invoke(input=messages)

    def astream(self, messages: List[BaseMessage]) -> Iterator[BaseMessageChunk]:
        raise NotImplementedError

    def stream(self, messages: List[BaseMessage]) -> Iterator[BaseMessageChunk]:
        raise NotImplementedError

    def batch(self, message_batches: List[List[BaseMessage]]) -> List[BaseMessage]:
        raise NotImplementedError

    async def abatch(
        self, message_batches: List[List[BaseMessage]]
    ) -> List[BaseMessage]:
        raise NotImplementedError
