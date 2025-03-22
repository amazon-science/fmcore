import random
from typing import List, Iterator

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, BaseMessageChunk

from fmcore.factory.bedrock_factory import (
    BedrockFactory,
    BedrockClientProxy,
)
from fmcore.llm.base_llm import BaseLLM
from fmcore.types.enums.provider_enums import ProviderType
from fmcore.types.llm_types import LLMConfig


class BedrockLLM(BaseLLM, BaseModel):
    """A language model implementation for AWS Bedrock service with built-in rate limiting and client management.

    This class manages multiple Bedrock clients with their individual rate limits and provides methods
    for both synchronous and asynchronous interactions. It uses weighted random selection based on
    rate limits when choosing a client for requests.

    Attributes:
        bedrock_clients (List[BedrockClientProxy]): List of rate-limited Bedrock client proxies.
        aliases (List[str]): Provider type aliases, set to [ProviderType.BEDROCK].
    """

    aliases = [ProviderType.BEDROCK]
    bedrock_clients: List[BedrockClientProxy] = Field(default_factory=list)

    @classmethod
    def _get_constructor_parameters(cls, *, llm_config: LLMConfig) -> dict:
        """Creates constructor parameters from the provided LLM configuration.

        Args:
            llm_config (LLMConfig): Configuration containing Bedrock account settings and model parameters.

        Returns:
            dict: Dictionary containing the config and initialized Bedrock clients.
        """
        bedrock_clients = BedrockFactory.create_bedrock_clients(llm_config=llm_config)
        return {"config": llm_config, "bedrock_clients": bedrock_clients}

    def get_random_client(self) -> BedrockClientProxy:
        """Selects a random Bedrock client using weighted random selection.

        The selection is weighted by each client's rate limit, giving higher probability
        to clients with higher rate limits. This helps distribute load optimally across
        clients with different capacities.

        Returns:
            BedrockClientProxy: A randomly selected client proxy.

        Raises:
            ValueError: If no Bedrock clients are available.
        """
        weights = [client.rate_limiter.max_rate for client in self.bedrock_clients]
        return random.choices(self.bedrock_clients, weights=weights, k=1)[0]

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """Synchronously invokes the Bedrock model with the given messages.

        Args:
            messages (List[BaseMessage]): The messages to send to the model.

        Returns:
            BaseMessage: The model's response.
        """
        bedrock_proxy: BedrockClientProxy = self.get_random_client()
        return bedrock_proxy.client.invoke(input=messages)

    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """Asynchronously invokes the Bedrock model with rate limiting.

        Args:
            messages (List[BaseMessage]): The messages to send to the model.

        Returns:
            BaseMessage: The model's response.

        Note:
            This method respects the rate limits of the selected client using an async context manager.
        """
        bedrock_proxy: BedrockClientProxy = self.get_random_client()
        async with bedrock_proxy.rate_limiter:
            return await bedrock_proxy.client.ainvoke(input=messages)

    def stream(self, messages: List[BaseMessage]) -> Iterator[BaseMessageChunk]:
        """Synchronously streams responses from the model.

        Args:
            messages (List[BaseMessage]): The messages to send to the model.

        Returns:
            Iterator[BaseMessageChunk]: An iterator of response chunks from the model.
        """
        bedrock_proxy: BedrockClientProxy = self.get_random_client()
        return bedrock_proxy.client.stream(input=messages)

    async def astream(self, messages: List[BaseMessage]) -> Iterator[BaseMessageChunk]:
        """Asynchronously streams responses from the model with rate limiting.

        Args:
            messages (List[BaseMessage]): The messages to send to the model.

        Returns:
            Iterator[BaseMessageChunk]: An iterator of response chunks from the model.

        Note:
            This method respects the rate limits of the selected client using an async context manager.
        """
        bedrock_proxy: BedrockClientProxy = self.get_random_client()
        async with bedrock_proxy.rate_limiter:
            return await bedrock_proxy.client.astream(input=messages)
