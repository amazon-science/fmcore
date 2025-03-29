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
    def _get_constructor_parameters(cls, *, llm_config: LLMConfig) -> dict:
        """
        Generate the constructor parameters required for initializing a subclass.

        This method is intended to be implemented by each subclass to provide the necessary parameters dynamically
        based on the given `llm_config`.The purpose of this abstraction is to ensure that the registry pattern used
        for creating subclasses does not require modification when new subclasses are introduced.

        Each subclass has a different set of constructor parameters, and this method allows them to generate those
        parameters as needed, ensuring flexibility and maintainability in the codebase.

        ---

        **How This Aligns with the Open/Closed Principle (OCP)**
        The **Open/Closed Principle** states that a system should be **open for extension but closed for modification**.
         This method enforces OCP by allowing new subclasses to be introduced **without modifying the base class** or
         the registry/factory mechanism. Instead of altering existing code when a new subclass is introduced, the new
         subclass simply implements `get_constructor_paramters`,ensuring it provides the appropriate constructor arguments.

        **Closed for Modification**:
        - The registry/factory does not need to be modified when adding a new subclass.
        - The base class remains unchanged, preventing regressions in existing functionality.

        **Open for Extension**:
        - New LLM subclasses can be introduced freely, each defining how to extract its own constructor parameters.
        - Different LLM implementations can have varying configurations, yet still be instantiated seamlessly within the existing system.

        ---

        Args:
            llm_config (LLMConfig): The configuration object containing LLM-related settings.

        Returns:
            dict: A dictionary of keyword arguments (`**kwargs`) that can be used to instantiate the subclass dynamically.

        By using this approach, we ensure that `BaseLLM` remains **stable and unmodified**, while allowing new
        subclasses to introduce their own custom constructor parameters without breaking existing functionality.
        """
        pass

    @classmethod
    def of(cls, llm_config: Union[LLMConfig, DistributedLLMConfig]):
        """
        Creates an instance of the appropriate LLM subclass based on the provided configuration.

        This method follows a hybrid of the Registry and Factory patterns:
        - Uses a registry to dynamically look up the correct subclass based on `config.provider`.
        - Acts as a factory by retrieving the necessary constructor parameters and instantiating the subclass.

        Args:
            llm_config (LLMConfig): The configuration containing provider details.

        Returns:
            BaseLLM: An instance of the corresponding LLM subclass.
        """

        # Hardcoding the "DistributedLLM" key to avoid a circular dependency.
        # If we introduce the DistributedLLM class here, it would require importing BaseLLM,
        # but BaseLLM also depends on DistributedLLM, causing a circular import error.
        key = (
            "DistributedLLM"
            if isinstance(llm_config, DistributedLLMConfig)
            else llm_config.provider_params.provider_type.name
        )

        BaseLLMClass = BaseLLM.get_subclass(key=key)
        constructor_params = BaseLLMClass._get_constructor_parameters(llm_config=llm_config)
        return BaseLLMClass(**constructor_params)

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
