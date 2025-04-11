import copy
from typing import List, Union

from fmcore.llm.types.llm_types import DistributedLLMConfig, LLMConfig
from fmcore.types.config_types import RateLimitConfig


class BandwidthDistributor:
    @staticmethod
    def distribute_bandwidth_across_processes_equally(
        num_process: int, llm_config: DistributedLLMConfig
    ) -> List[DistributedLLMConfig]:
        """
        Evenly distributes the total API rate limits defined in the input DistributedLLMConfig
        across a specified number of parallel processes.

        For each provider in the input config, its `max_rate` is divided among all processes.
        Each process receives a modified version of the provider configuration with a portion of
        the original rate limit. Any remainder is distributed among the first few processes.

        Only processes that receive a non-zero rate limit are included in the output.

        Args:
            num_process (int): Total number of parallel processes to distribute bandwidth across.
            llm_config (DistributedLLMConfig): The original configuration containing provider parameters
                                               with rate limits to be divided.

        Returns:
            List[DistributedLLMConfig]: A list of new DistributedLLMConfig instances, each containing
                                        a subset of the original provider parameters with updated
                                        rate limits. Only non-empty configurations are returned.
        """
        # Initialize empty lists for each process
        provider_params_list_per_process = [[] for _ in range(num_process)]

        # For each client config, split it across all processes
        for provider_params in llm_config.provider_params_list:
            # Calculate rate limit per process for this client config
            rate_limit_config: RateLimitConfig = provider_params.rate_limit
            rate_limit_per_process = rate_limit_config.max_rate // num_process
            # Calculate remainder to distribute
            remainder = rate_limit_config.max_rate % num_process

            # Distribute the client config across all processes
            for process_idx in range(num_process):
                # Add remainder to first few processes if there is any
                extra_rate = 1 if process_idx < remainder else 0
                assigned_rate = rate_limit_per_process + extra_rate

                if assigned_rate > 0:
                    provider_params_modified = provider_params.copy()
                    provider_params_modified.rate_limit.max_rate = assigned_rate
                    provider_params_list_per_process[process_idx].append(provider_params_modified)

        distributed_llm_configs: List[DistributedLLMConfig] = []
        for provider_params_list in provider_params_list_per_process:
            if provider_params_list:
                distributed_llm_config: DistributedLLMConfig = DistributedLLMConfig(
                    provider_type=llm_config.provider_type,
                    model_id=llm_config.model_id,
                    model_params=llm_config.model_params,
                    provider_params_list=provider_params_list,
                )
                distributed_llm_configs.append(distributed_llm_config)

        return distributed_llm_configs
