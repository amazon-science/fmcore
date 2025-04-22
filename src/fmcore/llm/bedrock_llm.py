from typing import List, Iterator, AsyncIterator

from aiolimiter import AsyncLimiter
from langchain_aws import ChatBedrockConverse
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, BaseMessageChunk

from fmcore.aws.constants.aws_constants import EXTENDED_THINKING_MODELS
from fmcore.aws.factory.bedrock_factory import BedrockFactory
from fmcore.llm.base_llm import BaseLLM
from fmcore.llm.types.llm_types import LLMConfig
from fmcore.utils.rate_limit_utils import RateLimiterUtils
from fmcore.utils.retry_utils import RetryUtil


class BedrockLLM(BaseLLM[List[BaseMessage], BaseMessage, BaseMessageChunk], BaseModel):
    """
    AWS Bedrock language model with built-in asynchronous rate limiting.

    This class encapsulates a ChatBedrockConverse client to enable both synchronous and
    asynchronous interactions with the Bedrock service. It is configured via an LLM configuration
    that includes model parameters and provider-specific settings.

    Attributes:
        client (ChatBedrockConverse): The underlying client for Bedrock conversations.
        rate_limiter (AsyncLimiter): Async rate limiter enforcing API rate limits.
    """

    aliases = ["BEDROCK"]

    client: ChatBedrockConverse
    rate_limiter: AsyncLimiter

    @classmethod
    def _validate_bedrock_params(cls, llm_config: LLMConfig) -> None:
        """
        Validates and adjusts thinking-related parameters for Bedrock models.

        Args:
            llm_config: The LLM configuration to validate

        Raises:
            ValueError: If token budget constraints are violated
        """
        model_params = llm_config.model_params

        if llm_config.model_id in EXTENDED_THINKING_MODELS:
            thinking_params = {}

            if (model_params.additional_model_request_fields is not None and
                    "thinking" in model_params.additional_model_request_fields):
                thinking_params = model_params.additional_model_request_fields["thinking"]

            if "type" in thinking_params:
                if thinking_params["type"] == "enabled":
                    # Fix temperature and unset top p for thinking mode
                    model_params.temperature = 1.0
                    model_params.top_p = None

                    # Validate token budgets
                    budget_tokens = thinking_params.get("budget_tokens", 0)
                    if budget_tokens < 1024:
                        raise ValueError("Budget tokens must be greater than 1024")
                    if budget_tokens > model_params.max_tokens:
                        raise ValueError("Max tokens must be greater than budget tokens")
                elif "budget_tokens" in thinking_params:
                    # Remove budget tokens if thinking is disabled
                    model_params.additional_model_request_fields["thinking"].pop("budget_tokens")
        else:
            # Clear additional fields for non-thinking models
            model_params.additional_model_request_fields = None

    @classmethod
    def _get_instance(cls, *, llm_config: LLMConfig) -> "BedrockLLM":
        """
        Constructs a BedrockLLM instance.

        Returns:
            - BedrockLLM: An implementation of BaseLLM.

        Args:
            llm_config (SingleLLMConfig): Contains model_id, model_params, and provider_params.
        """
        cls._validate_bedrock_params(llm_config)
        converse_client = BedrockFactory.create_converse_client(llm_config=llm_config)
        rate_limiter = RateLimiterUtils.create_async_rate_limiter(
            rate_limit_config=llm_config.provider_params.rate_limit
        )
        return BedrockLLM(config=llm_config, client=converse_client, rate_limiter=rate_limiter)

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Synchronously invokes the model with the given messages.

        Args:
            messages (List[BaseMessage]): The messages to send.

        Returns:
            BaseMessage: The model's response.
        """
        return self.client.invoke(input=messages)

    @RetryUtil.with_backoff(lambda self: self.config.provider_params.retries)
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

    @RetryUtil.with_backoff(lambda self: self.config.provider_params.retries)
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
