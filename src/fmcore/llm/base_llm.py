from abc import ABC, abstractmethod
from typing import Iterator, List, AsyncIterator, Union

from bears.util import Registry
from langchain_core.messages import BaseMessage, BaseMessageChunk

from fmcore.llm.types.llm_types import LLMConfig, DistributedLLMConfig
from fmcore.types.typed import MutableTyped


class BaseLLM(MutableTyped, Registry, ABC):
    """
    Abstract base class for LLM implementations.

    This class defines the interface and configuration for different LLMs.
    Concrete implementations must provide the actual logic for the abstract methods.

    Attributes:
        config (LLMConfig, DistributedLLMConfig): Configuration for the LLM.
    """
    config: Union[LLMConfig, DistributedLLMConfig]

    @classmethod
    @abstractmethod
    def _get_instance(cls, *, llm_config: LLMConfig) -> "BaseLLM":
        """
        Returns an instance of the llm subclass, initialized using the given `llm_config`.

        This method must be implemented by each subclass to construct and return an instance
        of itself. It enables dynamic instantiation of LLM implementations while keeping the
        base class and registry mechanism unchanged.

        This design supports the Open/Closed Principle by allowing new subclasses to define
        their own initialization logic without modifying the base infrastructure.

        Args:
            llm_config (LLMConfig): The configuration object containing settings for the LLM.

        Returns:
            BaseLLM: An instance of the subclass that extends `BaseLLM`.
        """

    @classmethod
    def of(cls, llm_config: Union[LLMConfig, DistributedLLMConfig]):
        """
        Instantiates an LLM subclass based on the provided configuration.

        This method acts as a centralized entry point for LLM instantiation. It leverages the
        registry mechanism to identify the correct LLM subclass using the `provider_type` from
        the configuration, and delegates construction to the subclass via `_get_instance`.

        This avoids tight coupling to specific implementations and supports runtime extensibility.

        Args:
            llm_config (Union[LLMConfig, DistributedLLMConfig]): The configuration object
                that defines which provider or distributed LLM to instantiate.

        Returns:
            BaseLLM: An instance of the resolved subclass implementing BaseLLM.
        """

        # Hardcoding the "DistributedLLM" key to avoid a circular dependency.
        # If we introduce the DistributedLLM class here, it would require importing BaseLLM,
        # but BaseLLM also depends on DistributedLLM, causing a circular import error.
        key = (
            "DistributedLLM"
            if isinstance(llm_config, DistributedLLMConfig)
            else llm_config.provider_type
        )

        BaseLLMClass = BaseLLM.get_subclass(key=key)
        return BaseLLMClass._get_instance(llm_config=llm_config)

    @abstractmethod
    def invoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Synchronously invokes the LLM with the given messages.

        Args:
            messages (List[BaseMessage]): The input messages.

        Returns:
            BaseMessage: The LLM response.
        """
        pass

    @abstractmethod
    async def ainvoke(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Asynchronously invokes the LLM with the given messages.

        Args:
            messages (List[BaseMessage]): The input messages.

        Returns:
            BaseMessage: The LLM response.
        """
        pass

    @abstractmethod
    def stream(self, messages: List[BaseMessage]) -> Iterator[BaseMessageChunk]:
        """
        Streams responses from the LLM for the given messages.

        Args:
            messages (List[BaseMessage]): The input messages.

        Returns:
            Iterator[BaseMessageChunk]: A stream of LLM response chunks.
        """
        pass

    @abstractmethod
    def astream(self, messages: List[BaseMessage]) -> AsyncIterator[BaseMessageChunk]:
        """
        Asynchronously streams responses from the LLM for the given messages.

        Args:
            messages (List[BaseMessage]): The input messages.

        Returns:
            Iterator[BaseMessageChunk]: A stream of LLM response chunks.
        """
        pass

    @abstractmethod
    def batch(self, messages: List[List[BaseMessage]]) -> List[BaseMessage]:
        """
        Processes a batch of messages in a single call.

        Args:
            messages (List[List[BaseMessage]]): A list of message lists.

        Returns:
            List[BaseMessage]: A list of responses corresponding to each input message list.
        """
        pass

    @abstractmethod
    async def abatch(self, messages: List[List[BaseMessage]]) -> List[BaseMessage]:
        """
        Asynchronously processes a batch of messages in a single call.

        Args:
            messages (List[List[BaseMessage]]): A list of message lists.

        Returns:
            List[BaseMessage]: A list of responses corresponding to each input message list.
        """
        pass
