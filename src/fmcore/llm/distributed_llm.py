import random
from typing import List, Iterator, AsyncIterator, Any

from fmcore.llm.base_llm import BaseLLM, Input, Output, Chunk
from fmcore.llm.types.llm_types import DistributedLLMConfig, LLMConfig


class DistributedLLM(BaseLLM[Input, Output, Chunk]):
    """
    Generic Distributed LLM that manages multiple LLM clients and distributes inference across them.

    This implementation allows for:
    - Supporting any type of LLM interface (custom input/output types)
    - Distributing load across multiple LLM instances/accounts
    - Weighted client selection based on rate limits

    Type Parameters:
        Input:  The type of input messages to the model
        Output: The type of output returned by the model
        Chunk:  The type of output chunk returned when streaming
    """

    config: DistributedLLMConfig
    llm_clients: List[BaseLLM[Input, Output, Chunk]]  # List of LLM clients, each with a unique account/config

    @classmethod
    def _get_instance(cls, *, llm_config: DistributedLLMConfig) -> "DistributedLLM[Input, Output, Chunk]":
        """
        Creates a DistributedLLM instance with multiple BaseLLM clients based on account-level settings.

        Each account in `provider_params_list` gets its own LLM client instance.

        Args:
            llm_config: Configuration object containing shared model info and per-account parameters.

        Returns:
            DistributedLLM: A fully initialized distributed client instance.
        """
        llm_clients = []

        # Create individual LLM clients per account configuration
        for provider_params in llm_config.provider_params_list:
            standalone_llm_config = LLMConfig(
                provider_type=llm_config.provider_type,
                model_id=llm_config.model_id,
                model_params=llm_config.model_params,
                provider_params=provider_params,  # Account-specific config
            )
            llm: BaseLLM[Input, Output, Chunk] = BaseLLM.of(llm_config=standalone_llm_config)
            llm_clients.append(llm)

        return cls(config=llm_config, llm_clients=llm_clients)

    def get_random_client(self) -> BaseLLM[Input, Output, Chunk]:
        """
        Selects a random LLM client for inference, weighted by its rate limit capacity.

        This ensures clients with higher throughput are utilized more frequently.

        Returns:
            BaseLLM: A randomly selected LLM client.
        """
        weights = [llm.rate_limiter.max_rate for llm in self.llm_clients]
        return random.choices(self.llm_clients, weights=weights, k=1)[0]

    def invoke(self, messages: Input) -> Output:
        """
        Synchronously invokes one of the distributed LLM clients.

        Args:
            messages: Input data (e.g., message list)

        Returns:
            Output: Response from the selected LLM client.
        """
        return self.get_random_client().invoke(messages)

    async def ainvoke(self, messages: Input) -> Output:
        """
        Asynchronously invokes one of the distributed LLM clients with rate limiting.

        Args:
            messages: Input data

        Returns:
            Output: Response from the selected LLM client.
        """
        return await self.get_random_client().ainvoke(messages)

    def stream(self, messages: Input) -> Iterator[Chunk]:
        """
        Synchronously streams the model's output in chunks.

        Args:
            messages: Input data

        Returns:
            Iterator[Chunk]: A streaming iterator over output chunks.
        """
        return self.get_random_client().stream(messages)

    async def astream(self, messages: Input) -> AsyncIterator[Chunk]:
        """
        Asynchronously streams output chunks from the model with rate limiting.

        Args:
            messages: Input data

        Returns:
            AsyncIterator[Chunk]: Asynchronous iterator over output chunks.
        """
        return await self.get_random_client().astream(messages)

    def batch(self, messages: List[Input]) -> List[Output]:
        """
        Synchronously performs batch inference using one LLM client.

        Args:
            messages: List of input items to process

        Returns:
            List[Output]: List of model outputs.
        """
        return self.get_random_client().batch(messages)

    async def abatch(self, messages: List[Input]) -> List[Output]:
        """
        Asynchronously performs batch inference with rate limiting.

        Args:
            messages: List of input items to process

        Returns:
            List[Output]: List of model outputs.
        """
        return await self.get_random_client().abatch(messages)
