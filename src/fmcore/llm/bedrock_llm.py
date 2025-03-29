from typing import List, Iterator, AsyncIterator

from aiolimiter import AsyncLimiter
from langchain_aws import ChatBedrockConverse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, BaseMessageChunk

from fmcore.aws.factory.bedrock_factory import BedrockFactory
from fmcore.llm.base_llm import BaseLLM
from fmcore.llm.enums.provider_enums import ProviderType
from fmcore.llm.types.llm_types import LLMConfig
from fmcore.utils.rate_limit_utils import RateLimiterUtils


class BedrockLLM(BaseLLM, BaseModel):
    """
    AWS Bedrock language model with built-in asynchronous rate limiting.

    This class encapsulates a ChatBedrockConverse client to enable both synchronous and
    asynchronous interactions with the Bedrock service. It is configured via an LLM configuration
    that includes model parameters and provider-specific settings.

    Attributes:
        client (ChatBedrockConverse): The underlying client for Bedrock conversations.
        rate_limiter (AsyncLimiter): Async rate limiter enforcing API rate limits.
    """

    aliases = [ProviderType.BEDROCK]

    client: ChatBedrockConverse
    rate_limiter: AsyncLimiter

    @classmethod
    def _get_constructor_parameters(cls, *, llm_config: LLMConfig) -> dict:
        """
        Constructs the initialization parameters for a BedrockLLM instance.

        Returns a dictionary containing:
            - config: The original LLM configuration.
            - client: A ChatBedrockConverse client built from the configuration.
            - rate_limiter: An AsyncLimiter based on the provider's rate limit settings.

        Args:
            llm_config (SingleLLMConfig): Contains model_id, model_params, and provider_params.
        """
        converse_client = BedrockFactory.create_converse_client(llm_config=llm_config)
        rate_limiter = RateLimiterUtils.create_async_rate_limiter(
            rate_limit_config=llm_config.provider_params.rate_limit
        )
        return {"config": llm_config, "client": converse_client, "rate_limiter": rate_limiter}

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Synchronously invokes the model with the given messages.

        Args:
            messages (List[BaseMessage]): The messages to send.

        Returns:
            BaseMessage: The model's response.
        """
        return self.client.invoke(input=messages)

    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Asynchronously invokes the model with rate limiting.

        Args:
            messages (List[BaseMessage]): The messages to send.

        Returns:
            BaseMessage: The model's response.
        """
        async with self.rate_limiter:
            return await self.client.ainvoke(input=messages)

    def stream(self, messages: List[BaseMessage]) -> Iterator[BaseMessageChunk]:
        """
        Synchronously streams response chunks from the model.

        Args:
            messages (List[BaseMessage]): The messages to send.

        Returns:
            Iterator[BaseMessageChunk]: An iterator over response chunks.
        """
        return self.client.stream(input=messages)

    async def astream(self, messages: List[BaseMessage]) -> AsyncIterator[BaseMessageChunk]:
        """
        Asynchronously streams response chunks from the model with rate limiting.

        Args:
            messages (List[BaseMessage]): The messages to send.

        Returns:
            AsyncIterator[BaseMessageChunk]: An async iterator over response chunks.
        """
        async with self.rate_limiter:
            return self.client.astream(input=messages)

    def batch(self, messages: List[List[BaseMessage]]) -> List[BaseMessage]:
        raise NotImplementedError("Batch processing is not implemented for BedrockLLM.")

    async def abatch(self, messages: List[List[BaseMessage]]) -> List[BaseMessage]:
        raise NotImplementedError("Batch processing is not implemented for BedrockLLM.")
