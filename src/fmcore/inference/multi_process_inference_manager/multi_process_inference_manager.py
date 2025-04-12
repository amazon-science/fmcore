import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List

from langchain_core.messages import BaseMessage
from tqdm import tqdm

from fmcore.inference.base_inference_manager import BaseInferenceManager, I, O
from fmcore.inference.multi_process_inference_manager.bandwidth_distributor import BandwidthDistributor
from fmcore.inference.multi_process_inference_manager.types.multi_process_inference_manager_types import (
    MultiProcessWorkerConfig,
    MultiProcessInferenceManagerParams,
)
from fmcore.inference.types.inference_manager_types import InferenceManagerConfig
from fmcore.llm import BaseLLM
from fmcore.llm.types.llm_types import LLMConfig, DistributedLLMConfig
from fmcore.utils.collection_utils import CollectionUtils
from fmcore.utils.logging_utils import Log


class MultiProcessInferenceManager(BaseInferenceManager[List[List[BaseMessage]], List[BaseMessage]]):
    """
    Inference manager that distributes workload across multiple processes for parallel execution.

    This class is designed to perform inference on large datasets using multiprocessing,
    where each process operates on a subset (chunk) of the data using its own LLM configuration.
    It leverages asynchronous processing within each process and merges the results after completion.

    Attributes:
        aliases (List[str]): List of identifiers used to reference this inference manager.
    """

    aliases = ["MULTI_PROCESS"]

    @classmethod
    def _get_instance(cls, *, config: InferenceManagerConfig) -> "BaseInferenceManager":
        return cls(config=config)

    async def process_chunk(self, worker_config: MultiProcessWorkerConfig) -> List[BaseMessage]:
        """
        Asynchronously process a chunk of messages using the provided LLM config.

        Args:
            worker_config (WorkerConfig): Configuration for this chunk.

        Returns:
            List[BaseMessage]: Responses generated for each message group in the chunk.
        """
        llm = BaseLLM.of(llm_config=worker_config.llm_config)
        tasks = [llm.ainvoke(messages=messages) for messages in worker_config.dataset_chunk]

        results = []
        for task in tqdm(tasks, total=len(tasks), desc=f"Processing chunk {worker_config.chunk_id}"
        ):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                Log.info(f"Error processing chunk {worker_config.chunk_id}: {e}, Gracefully ignoring the row")

        return results

    def worker(self, worker_config: MultiProcessWorkerConfig) -> List[BaseMessage]:
        """
        Entry point for multiprocessing worker.

        Creates and runs a new event loop to process the given chunk.

        Args:
            worker_config (WorkerConfig): Configuration for this chunk.

        Returns:
            List[BaseMessage]: List of generated model responses.
        """
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(self.process_chunk(worker_config))
        finally:
            loop.close()

    def run(self, dataset: List[BaseMessage]) -> List[BaseMessage]:
        """
        Run inference on the given dataset using multiple processes.

        Splits the dataset into balanced chunks, assigns each chunk a bandwidth-optimized
        LLM configuration, and executes them in parallel using a thread pool.

        Args:
            dataset (List[BaseMessage]): Dataset to be processed.

        Returns:
            List[BaseMessage]: Flattened list of all model responses from each chunk.
        """
        inference_params: MultiProcessInferenceManagerParams = self.config.inference_manager_params
        num_process = inference_params.num_process

        configs: List[DistributedLLMConfig] = (
            BandwidthDistributor.distribute_bandwidth_across_processes_equally(
                num_process=num_process,
                llm_config=self.config.llm_config,
            )
        )

        num_process: int = min(num_process, len(configs))
        chunks: List[List[BaseMessage]] = CollectionUtils.split_into_equal_parts(
            items=dataset, num_parts=num_process
        )

        result_list: List[List[BaseMessage]] = []

        with ThreadPoolExecutor(max_workers=num_process) as executor:
            future_list = []
            for chunk_id, (chunk, llm_config) in enumerate(zip(chunks, configs)):
                worker_config: MultiProcessWorkerConfig = MultiProcessWorkerConfig(
                    chunk_id=chunk_id, dataset_chunk=chunk, llm_config=llm_config
                )
                future = executor.submit(self.worker, worker_config)
                future_list.append(future)

            for future in tqdm(future_list, total=len(future_list), desc="Executing Chunk..."):
                result_list.append(future.result())

        all_results: List[BaseMessage] = [item for sublist in result_list for item in sublist]

        return all_results
