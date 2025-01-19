import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures._base import Executor
from typing import *

import numpy as np
from pydantic import Extra, root_validator

from fmcore.constants.DataProcessingConstants import Parallelize
from fmcore.util.language import (
    Alias,
    PandasSeries,
    Parameters,
    ProgressBar,
    filter_kwargs,
    get_default,
    is_dict_like,
    is_list_or_set_like,
    set_param_from_alias,
    type_str,
)

from ._asyncio import run_asyncio
from ._processes import ActorPoolExecutor, ActorProxy, run_parallel
from ._ray import RayPoolExecutor, run_parallel_ray
from ._threads import (
    RestrictedConcurrencyThreadPoolExecutor,
    kill_thread,
    run_concurrent,
    suppress_ThreadKilledSystemException,
)
from ._utils import (
    _LOCAL_ACCUMULATE_ITEM_WAIT,
    _LOCAL_ACCUMULATE_ITER_WAIT,
    _RAY_ACCUMULATE_ITEM_WAIT,
    _RAY_ACCUMULATE_ITER_WAIT,
    accumulate,
    accumulate_iter,
)


def worker_ids(executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor, ActorPoolExecutor]]) -> Set[int]:
    ## Returns a set of unique identifiers for all workers in the given executor
    ## Input: executor - any supported pool executor (Thread, Process, or Actor)
    ## Output: Set of thread IDs or process IDs depending on executor type
    
    if isinstance(executor, ThreadPoolExecutor):
        ## For thread pools, return set of thread identifiers
        return {th.ident for th in executor._threads}
    elif isinstance(executor, ProcessPoolExecutor):
        ## For process pools, return set of process IDs
        return {p.pid for p in executor._processes.values()}
    elif isinstance(executor, ActorPoolExecutor):
        ## For actor pools, return set of actor process IDs
        return {_actor._process.pid for _actor in executor._actors}
    
    ## Raise error if executor type is not supported
    raise NotImplementedError(f'Cannot get worker ids for executor of type: {executor}')


class ExecutorConfig(Parameters):
    """
    Configuration class for parallel execution settings used by dispatch functions.
    Provides a structured way to define parallelization strategy and execution constraints.

    Attributes:
        parallelize: Type of parallelization to use (sync, threads, processes, ray)
        max_workers: Maximum number of parallel workers (None uses system defaults)
        max_calls_per_second: Rate limiting for execution calls (infinity means no limit)

    Example usage:
        >>> config = ExecutorConfig(
                parallelize='threads',
                max_workers=4,
                max_calls_per_second=100.0
            )
        >>> executor = dispatch_executor(config=config)
        
        # Using with num_workers alias
        >>> config = ExecutorConfig(
                parallelize='processes',
                num_workers=8  # alias for max_workers
            )
    """
    class Config(Parameters.Config):
        extra = Extra.ignore  ## Silently ignore any extra parameters for flexibility

    parallelize: Parallelize
    max_workers: Optional[int] = None  ## None lets the executor use system-appropriate defaults
    max_calls_per_second: float = float('inf')  ## No rate limiting by default

    @root_validator(pre=True)
    def _set_params(cls, params: Dict) -> Dict:
        """
        Pre-processes configuration parameters to support alternate parameter names.
        Allows 'num_workers' as an alias for 'max_workers' for compatibility.
        """
        set_param_from_alias(params, param='max_workers', alias=['num_workers'], default=None)
        return params


def dispatch(
        fn: Callable,
        *args,
        parallelize: Parallelize,
        forward_parallelize: bool = False,
        delay: float = 0.0,
        executor: Optional[Executor] = None,
        **kwargs
) -> Any:
    parallelize: Parallelize = Parallelize.from_str(parallelize)
    if forward_parallelize:
        kwargs['parallelize'] = parallelize
    time.sleep(delay)
    if parallelize is Parallelize.sync:
        return fn(*args, **kwargs)
    elif parallelize is Parallelize.asyncio:
        return run_asyncio(fn, *args, **kwargs)
    elif parallelize is Parallelize.threads:
        return run_concurrent(fn, *args, executor=executor, **kwargs)
    elif parallelize is Parallelize.processes:
        return run_parallel(fn, *args, executor=executor, **kwargs)
    elif parallelize is Parallelize.ray:
        return run_parallel_ray(fn, *args, executor=executor, **kwargs)
    raise NotImplementedError(f'Unsupported parallelization: {parallelize}')


def dispatch_executor(
        *,
        config: Optional[Union[ExecutorConfig, Dict]] = None,
        **kwargs
) -> Optional[Executor]:
    """
    Creates and configures an executor based on the provided configuration settings.
    Returns None for synchronous execution or when using default system executors.
    
    The executor handles parallel task execution with configurable constraints like
    maximum workers and rate limiting for thread-based execution.

    Args:
        config: ExecutorConfig instance or dict containing parallelization settings
        **kwargs: Additional configuration parameters that override config values

    Returns:
        Configured executor instance or None if using defaults/sync execution

    Example usage:
        >>> config = ExecutorConfig(
                parallelize='threads',
                max_workers=4,
                max_calls_per_second=100.0
            )
        >>> executor = dispatch_executor(config=config)
        
        >>> executor = dispatch_executor(
                config=dict(parallelize='processes', max_workers=8)
            )
    """
    if config is None:
        config: Dict = dict()
    else:
        assert isinstance(config, ExecutorConfig)
        config: Dict = config.dict(exclude=True)
    
    ## Merge passed kwargs with config dict to allow parameter overrides
    config: ExecutorConfig = ExecutorConfig(**{**config, **kwargs})

    if config.max_workers is None:
        ## Return None to use system defaults - this is more efficient for simple cases
        return None

    if config.parallelize is Parallelize.sync:
        return None
    elif config.parallelize is Parallelize.threads:
        ## Use restricted concurrency for threads to enable rate limiting
        return RestrictedConcurrencyThreadPoolExecutor(
            max_workers=config.max_workers,
            max_calls_per_second=config.max_calls_per_second,
        )
    elif config.parallelize is Parallelize.processes:
        ## Actor-based pool enables better control over process lifecycle
        return ActorPoolExecutor(
            max_workers=config.max_workers,
        )
    elif config.parallelize is Parallelize.ray:
        ## Ray executor for distributed execution across multiple machines
        return RayPoolExecutor(
            max_workers=config.max_workers,
        )
    else:
        raise NotImplementedError(f'Unsupported: you cannot create an executor with {config.parallelize} parallelization.')


def dispatch_apply(
        struct: Union[List, Tuple, np.ndarray, PandasSeries, Set, frozenset, Dict],
        *args,
        fn: Callable,
        parallelize: Parallelize,
        forward_parallelize: bool = False,
        item_wait: Optional[float] = None,
        iter_wait: Optional[float] = None,
        iter: bool = False,
        **kwargs
) -> Any:
    """
    Applies a function to each element in a data structure in parallel using the specified execution strategy.
    Similar to map() but with parallel execution capabilities and progress tracking.

    The function handles different types of parallel execution:
    - Synchronous (single-threaded)
    - Asyncio-based concurrent execution (for low-latency async/await functions)
    - Thread-based parallelism (for IO-bound tasks)
    - Process-based parallelism (for CPU-bound tasks)
    - Ray-based distributed execution (for multi-machine execution)

    Args:
        struct: Input data structure to iterate over. Can be list-like or dict-like
        *args: Additional positional args passed to each fn call
        fn: Function to apply to each element
        parallelize: Execution strategy (sync, threads, processes, ray, asyncio)
        forward_parallelize: If True, passes the parallelize strategy to fn
        item_wait: Delay between submitting individual items (rate limiting)
        iter_wait: Delay between checking completion of submitted items
        iter: If True, returns an iterator that yields results as they complete
        **kwargs: Additional keyword args passed to each fn call

    Example usage:
        >>> data = [1, 2, 3, 4, 5]
        >>> def square(x):
                return x * x
        
        >>> ## Process items in parallel using threads
        >>> results = dispatch_apply(
                data,
                fn=square, 
                parallelize='threads',
                max_workers=4
            )

        >>> ## Process dictionary items using processes
        >>> data = {'a': 1, 'b': 2, 'c': 3}
        >>> results = dispatch_apply(
                data,
                fn=square,
                parallelize='processes',
                progress_bar=True
            )
    """
    ## Convert string parallelization strategy to enum
    parallelize: Parallelize = Parallelize.from_str(parallelize)

    ## Set appropriate wait times based on execution strategy:
    ## - Sync/asyncio don't need waits since they're single-threaded
    ## - Local execution (threads/processes) can use shorter waits
    ## - Ray execution needs longer waits due to distributed nature
    item_wait: float = get_default(
        item_wait,
        {
            Parallelize.ray: _RAY_ACCUMULATE_ITEM_WAIT,
            Parallelize.processes: _LOCAL_ACCUMULATE_ITEM_WAIT,
            Parallelize.threads: _LOCAL_ACCUMULATE_ITEM_WAIT,
            Parallelize.asyncio: 0.0,
            Parallelize.sync: 0.0,
        }[parallelize]
    )
    iter_wait: float = get_default(
        iter_wait,
        {
            Parallelize.ray: _RAY_ACCUMULATE_ITER_WAIT,
            Parallelize.processes: _LOCAL_ACCUMULATE_ITER_WAIT,
            Parallelize.threads: _LOCAL_ACCUMULATE_ITER_WAIT,
            Parallelize.asyncio: 0.0,
            Parallelize.sync: 0.0,
        }[parallelize]
    )

    ## Forward parallelization strategy to child function if requested:
    if forward_parallelize:
        kwargs['parallelize'] = parallelize

    ## Create appropriate executor based on parallelization strategy:
    executor: Optional = dispatch_executor(
        parallelize=parallelize,
        **kwargs,
    )

    try:
        ## Configure progress bars for both submission and collection phases.
        ## Default to showing progress unless explicitly disabled:
        progress_bar: Optional[Dict] = Alias.get_progress_bar(kwargs)
        submit_pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(struct),
            desc='Submitting',
            prefer_kwargs=False,
            unit='item',
        )
        collect_pbar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=len(struct),
            desc='Collecting',
            prefer_kwargs=False,
            unit='item',
        )

        ## Handle list-like structures (lists, tuples, sets, arrays):
        if is_list_or_set_like(struct):
            futs = []
            for v in struct:
                ## Wrap user function to handle item-level execution
                def submit_task(item, **dispatch_kwargs):
                    return fn(item, **dispatch_kwargs)

                ## Submit task for parallel execution with rate limiting (item_wait):
                futs.append(
                    dispatch(
                        fn=submit_task,
                        item=v,
                        parallelize=parallelize,
                        executor=executor,
                        delay=item_wait,
                        **filter_kwargs(fn, **kwargs),
                    )
                )
                submit_pbar.update(1)

        ## Handle dictionary-like structures:
        elif is_dict_like(struct):
            futs = {}
            for k, v in struct.items():
                def submit_task(item, **dispatch_kwargs):
                    return fn(item, **dispatch_kwargs)

                ## Submit task with key for maintaining dict structure:
                futs[k] = dispatch(
                    fn=submit_task,
                    key=k,
                    item=v,
                    parallelize=parallelize,
                    executor=executor,
                    delay=item_wait,
                    **filter_kwargs(fn, **kwargs),
                )
                submit_pbar.update(1)
        else:
            raise NotImplementedError(f'Unsupported type: {type_str(struct)}')

        submit_pbar.success()

        ## Return results either as iterator or all-at-once (afer accumulating all futures):
        if iter:
            return accumulate_iter(
                futs,
                item_wait=item_wait,
                iter_wait=iter_wait,
                progress_bar=collect_pbar,
                **kwargs
            )
        else:
            return accumulate(
                futs,
                item_wait=item_wait,
                iter_wait=iter_wait,
                progress_bar=collect_pbar,
                **kwargs
            )
    finally:
        ## Ensure executor is properly cleaned up even if processing fails:
        stop_executor(executor)


def stop_executor(
        executor: Optional[Executor],
        force: bool = True,  ## Forcefully terminate, might lead to work being lost.
):
    if executor is not None:
        if isinstance(executor, ThreadPoolExecutor):
            suppress_ThreadKilledSystemException()
            if force:
                executor.shutdown(wait=False)  ## Cancels pending items
                for tid in worker_ids(executor):
                    kill_thread(tid)  ## Note; after calling this, you can still submit
                executor.shutdown(wait=False)  ## Note; after calling this, you cannot submit
            else:
                executor.shutdown(wait=True)
            del executor
        elif isinstance(executor, ProcessPoolExecutor):
            if force:
                for process in executor._processes.values():  # Internal Process objects
                    process.terminate()  # Forcefully terminate the process

                # Wait for the processes to clean up
                for process in executor._processes.values():
                    process.join()
                executor.shutdown(wait=True, cancel_futures=True)
            else:
                executor.shutdown(wait=True, cancel_futures=True)
            del executor
        elif isinstance(executor, ActorPoolExecutor):
            for actor in executor._actors:
                assert isinstance(actor, ActorProxy)
                actor.stop(cancel_futures=force)
                del actor
            del executor
