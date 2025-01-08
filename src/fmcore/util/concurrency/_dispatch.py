"""A collection of concurrency utilities to augment the Python language:"""
## Jupyter-compatible asyncio usage:
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures._base import Executor
from typing import *

import numpy as np
from pydantic import Extra, root_validator

from fmcore.constants.DataProcessingConstants import Parallelize
from fmcore.util.language import ProgressBar, set_param_from_alias, type_str, get_default, Parameters, \
    is_list_or_set_like, is_dict_like, PandasSeries, filter_kwargs
from ._asyncio import run_asyncio
from ._processes import ActorPoolExecutor, ActorProxy, run_parallel
from ._ray import RayPoolExecutor, run_parallel_ray
from ._threads import suppress_ThreadKilledSystemException, kill_thread, RestrictedConcurrencyThreadPoolExecutor, run_concurrent
from ._utils import accumulate_iter, accumulate, \
    _RAY_ACCUMULATE_ITEM_WAIT, _RAY_ACCUMULATE_ITER_WAIT, _LOCAL_ACCUMULATE_ITEM_WAIT, _LOCAL_ACCUMULATE_ITER_WAIT


def worker_ids(executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor, ActorPoolExecutor]]) -> Set[int]:
    if isinstance(executor, ThreadPoolExecutor):
        return {th.ident for th in executor._threads}
    elif isinstance(executor, ProcessPoolExecutor):
        return {p.pid for p in executor._processes.values()}
    elif isinstance(executor, ActorPoolExecutor):
        return {_actor._process.pid for _actor in executor._actors}
    raise NotImplementedError(f'Cannot get worker ids for executor of type: {executor}')


class ExecutorConfig(Parameters):
    class Config(Parameters.Config):
        extra = Extra.ignore

    parallelize: Parallelize
    max_workers: Optional[int] = None
    max_calls_per_second: float = float('inf')

    @root_validator(pre=True)
    def _set_params(cls, params: Dict) -> Dict:
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
    if config is None:
        config: Dict = dict()
    else:
        assert isinstance(config, ExecutorConfig)
        config: Dict = config.dict(exclude=True)
    config: ExecutorConfig = ExecutorConfig(**{**config, **kwargs})
    if config.max_workers is None:
        ## Uses the default executor for threads/processes/ray.
        return None
    if config.parallelize is Parallelize.sync:
        return None
    elif config.parallelize is Parallelize.threads:
        return RestrictedConcurrencyThreadPoolExecutor(
            max_workers=config.max_workers,
            max_calls_per_second=config.max_calls_per_second,
        )
    elif config.parallelize is Parallelize.processes:
        return ActorPoolExecutor(
            max_workers=config.max_workers,
        )
    elif config.parallelize is Parallelize.ray:
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
    parallelize: Parallelize = Parallelize.from_str(parallelize)
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
    if forward_parallelize:
        kwargs['parallelize'] = parallelize
    executor: Optional = dispatch_executor(
        parallelize=parallelize,
        **kwargs,
    )
    try:
        set_param_from_alias(kwargs, param='progress_bar', alias=['progress', 'pbar'], default=True)
        progress_bar: Union[ProgressBar, Dict, bool] = kwargs.pop('progress_bar', False)
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
        if is_list_or_set_like(struct):
            futs = []
            for v in struct:
                def submit_task(item, **dispatch_kwargs):
                    return fn(item, **dispatch_kwargs)

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
        elif is_dict_like(struct):
            futs = {}
            for k, v in struct.items():
                def submit_task(item, **dispatch_kwargs):
                    return fn(item, **dispatch_kwargs)

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
