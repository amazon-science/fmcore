"""A collection of concurrency utilities to augment the Python language:"""
## Jupyter-compatible asyncio usage:
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import Executor
from math import inf
from typing import *

import ray
from ray.util.dask import RayDaskCallback

from fmcore.util.language import Parameters
from ._utils import is_done, wait, _RAY_ACCUMULATE_ITER_WAIT, _RAY_ACCUMULATE_ITEM_WAIT


@ray.remote(num_cpus=1)
def _run_parallel_ray_executor(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _ray_asyncio_start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


class RayPoolExecutor(Executor, Parameters):
    max_workers: Union[int, Literal[inf]]
    iter_wait: float = _RAY_ACCUMULATE_ITER_WAIT
    item_wait: float = _RAY_ACCUMULATE_ITEM_WAIT
    _asyncio_event_loop: Optional = None
    _asyncio_event_loop_thread: Optional = None
    _submission_executor: Optional[ThreadPoolExecutor] = None
    _running_tasks: Dict = {}
    _latest_submit: Optional[int] = None

    def _set_asyncio(self):
        # Create a new loop and a thread running this loop
        if self._asyncio_event_loop is None:
            self._asyncio_event_loop = asyncio.new_event_loop()
            # print(f'Started _asyncio_event_loop')
        if self._asyncio_event_loop_thread is None:
            self._asyncio_event_loop_thread = threading.Thread(
                target=_ray_asyncio_start_event_loop,
                args=(self._asyncio_event_loop,),
            )
            self._asyncio_event_loop_thread.start()
            # print(f'Started _asyncio_event_loop_thread')

    def submit(
            self,
            fn,
            *args,
            scheduling_strategy: str = "SPREAD",
            num_cpus: int = 1,
            num_gpus: int = 0,
            max_retries: int = 0,
            retry_exceptions: Union[List, bool] = True,
            **kwargs,
    ):
        # print(f'Running {fn_str(fn)} using {Parallelize.ray} with num_cpus={num_cpus}, num_gpus={num_gpus}')
        def _submit_task():
            return _run_parallel_ray_executor.options(
                scheduling_strategy=scheduling_strategy,
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                max_retries=max_retries,
                retry_exceptions=retry_exceptions,
            ).remote(fn, *args, **kwargs)

        _task_uid = str(time.time_ns())

        if self.max_workers == inf:
            return _submit_task()  ## Submit to Ray directly
        self._set_asyncio()
        ## Create a coroutine (i.e. Future), but do not actually start executing it.
        coroutine = self._ray_run_fn_async(
            submit_task=_submit_task,
            task_uid=_task_uid,
        )

        ## Schedule the coroutine to execute on the event loop (which is running on thread _asyncio_event_loop).
        fut = asyncio.run_coroutine_threadsafe(coroutine, self._asyncio_event_loop)
        # while _task_uid not in self._running_tasks:  ## Ensure task has started scheduling
        #     time.sleep(self.item_wait)
        return fut

    async def _ray_run_fn_async(
            self,
            submit_task: Callable,
            task_uid: str,
    ):
        # self._running_tasks[task_uid] = None
        while len(self._running_tasks) >= self.max_workers:
            for _task_uid in sorted(self._running_tasks.keys()):
                if is_done(self._running_tasks[_task_uid]):
                    self._running_tasks.pop(_task_uid, None)
                    # print(f'Popped {_task_uid}')
                    if len(self._running_tasks) < self.max_workers:
                        break
                time.sleep(self.item_wait)
            if len(self._running_tasks) < self.max_workers:
                break
            time.sleep(self.iter_wait)
        fut = submit_task()
        self._running_tasks[task_uid] = fut
        # print(f'Started {task_uid}. Num running: {len(self._running_tasks)}')

        # ## Cleanup any completed tasks:
        # for k in list(self._running_tasks.keys()):
        #     if is_done(self._running_tasks[k]):
        #         self._running_tasks.pop(k, None)
        #     time.sleep(self.item_wait)
        return fut


def run_parallel_ray(
        fn,
        *args,
        scheduling_strategy: str = "SPREAD",
        num_cpus: int = 1,
        num_gpus: int = 0,
        max_retries: int = 0,
        retry_exceptions: Union[List, bool] = True,
        executor: Optional[RayPoolExecutor] = None,
        **kwargs,
):
    # print(f'Running {fn_str(fn)} using {Parallelize.ray} with num_cpus={num_cpus}, num_gpus={num_gpus}')
    if executor is not None:
        assert isinstance(executor, RayPoolExecutor)
        return executor.submit(
            fn,
            *args,
            scheduling_strategy=scheduling_strategy,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
            **kwargs,
        )
    else:
        return _run_parallel_ray_executor.options(
            scheduling_strategy=scheduling_strategy,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
        ).remote(fn, *args, **kwargs)


## Ref: https://docs.ray.io/en/latest/data/dask-on-ray.html#callbacks
class RayDaskPersistWaitCallback(RayDaskCallback):
    ## Callback to wait for computation to complete when .persist() is called with block=True
    def _ray_postsubmit_all(self, object_refs, dsk):
        wait(object_refs)
