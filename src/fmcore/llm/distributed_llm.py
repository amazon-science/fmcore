import random
from typing import List, Iterator, AsyncIterator

from langchain_core.messages import BaseMessage, BaseMessageChunk

from fmcore.llm.base_llm import BaseLLM
from fmcore.types.llm_types import DistributedLLMConfig, LLMConfig


class DistributedLLM(BaseLLM):
    """
    Distributed LLM implementation that manages multiple LLM clients.

    This class initializes multiple LLM instances based on the provided configuration,
    allowing distributed requests across multiple accounts.

    Attributes:
        llm_clients (List[BaseLLM]): A list of LLM instances, each associated with a different account.
    """

    llm_clients: List[BaseLLM]

    @classmethod
    def _get_constructor_parameters(cls, *, llm_config: DistributedLLMConfig) -> dict:
        """
        Creates initialization parameters for the DistributedLLM.

        This method initializes an LLM instance for each account in the configuration.

        Args:
            llm_config (DistributedLLMConfig): Configuration containing model details
                                               and account-specific settings.

        Returns:
            dict: A dictionary containing:
                - "config": The original LLM configuration.
                - "llms": A list of LLM instances, one for each account.
        """

        llm_clients = []
        for provider_params in llm_config.provider_params_list:
            standalone_llm_config = LLMConfig(
                model_id=llm_config.model_id,
                model_params=llm_config.model_params,
                provider_params=provider_params,  # Using individual account settings
            )
            llm: BaseLLM = BaseLLM.of(llm_config=standalone_llm_config)
            llm_clients.append(llm)

        return {"config": llm_config, "llm_clients": llm_clients}

    def get_random_client(self) -> BaseLLM:
        weights = [llm.rate_limiter.max_rate for llm in self.llm_clients]
        return random.choices(self.llm_clients, weights=weights, k=1)[0]

    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """Synchronously invokes the Bedrock model with the given messages.
        Args:
            messages (List[BaseMessage]): The messages to send to the model.
        Returns:
            BaseMessage: The model's response.
        """
        llm: BaseLLM = self.get_random_client()
        return llm.invoke(messages=messages)

    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """Asynchronously invokes the Bedrock model with rate limiting.
        Args:
            messages (List[BaseMessage]): The messages to send to the model.
        Returns:
            BaseMessage: The model's response.
        Note:
            This method respects the rate limits of the selected client using an async context manager.
        """
        llm: BaseLLM = self.get_random_client()
        return await llm.ainvoke(messages=messages)

    def stream(self, messages: List[BaseMessage]) -> Iterator[BaseMessageChunk]:
        """Synchronously streams responses from the model.
        Args:
            messages (List[BaseMessage]): The messages to send to the model.
        Returns:
            Iterator[BaseMessageChunk]: An iterator of response chunks from the model.
        """
        llm: BaseLLM = self.get_random_client()
        return llm.stream(messages=messages)

    async def astream(self, messages: List[BaseMessage]) -> AsyncIterator[BaseMessageChunk]:
        """Asynchronously streams responses from the model with rate limiting.
        Args:
            messages (List[BaseMessage]): The messages to send to the model.
        Returns:
            Iterator[BaseMessageChunk]: An iterator of response chunks from the model.
        Note:
            This method respects the rate limits of the selected client using an async context manager.
        """
        llm: BaseLLM = self.get_random_client()
        return await llm.astream(messages=messages)
