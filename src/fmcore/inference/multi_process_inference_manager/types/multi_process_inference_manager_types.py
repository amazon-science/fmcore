import multiprocessing as mp
from typing import List, Any

from pydantic import Field

from fmcore.inference.types.inference_manager_types import InferenceManagerParams
from fmcore.llm.types.llm_types import LLMConfig, DistributedLLMConfig
from fmcore.types.typed import MutableTyped


class MultiProcessInferenceManagerParams(InferenceManagerParams):
    """
    Configuration parameters specific to the multi-process inference manager.

    Attributes:
        num_process (int): Number of parallel processes to spawn. Defaults to the number of CPU cores.
        distribution_strategy (str): Strategy for distributing load across processes. Defaults to "uniform".

    Class Attributes:
        aliases (List[str]): List of aliases to identify this config type in the registry.
    """

    aliases = ["MULTI_PROCESS"]
    num_process: int = Field(default=mp.cpu_count())
    distribution_strategy: str = Field(default="uniform")


class MultiProcessWorkerConfig(MutableTyped):
    """
    Configuration object passed to each worker process during multi-process inference.

    Attributes:
        chunk_id (int): Unique identifier for the chunk being processed.
        dataset_chunk (Any): Subset of the dataset to be processed by this worker.
        llm_config (DistributedLLMConfig): LLM configuration specific to this worker.
    """

    chunk_id: int
    dataset_chunk: Any
    llm_config: DistributedLLMConfig
