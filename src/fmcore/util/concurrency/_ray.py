import asyncio
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import Executor
from contextlib import contextmanager
from math import inf
from typing import *

from pydantic import Extra, confloat, conint

from fmcore.util.language import (
    Alias,
    Parameters,
    ProgressBar,
    String,
    UserEnteredParameters,
    as_list,
)
from fmcore.util.language._import import (
    _IS_DASK_INSTALLED,
    _IS_RAY_INSTALLED,
    _check_is_ray_installed,
)

from ._utils import (
    _RAY_ACCUMULATE_ITEM_WAIT,
    _RAY_ACCUMULATE_ITER_WAIT,
    get_result,
    is_done,
    wait,
)

RayRuntimeEnv = dict
RequestCounter = 'RequestCounter'
if _IS_RAY_INSTALLED:
    import ray
    from ray.runtime_env import RuntimeEnv as RayRuntimeEnv


    @ray.remote(num_cpus=1)
    def _run_parallel_ray_executor(fn, *args, **kwargs):
        return fn(*args, **kwargs)


    @ray.remote
    class RequestCounter:
        def __init__(self):
            self.pending_requests: int = 0
            self.last_started: float = -1
            self.last_completed: float = -1

        def started_request(self):
            self.pending_requests += 1
            self.last_started: time.time()

        def completed_request(self):
            self.pending_requests -= 1
            self.last_completed: time.time()

        def num_pending_requests(self) -> int:
            return self.pending_requests

        def last_started_timestamp(self) -> float:
            return self.last_started

        def last_completed_timestamp(self) -> float:
            return self.last_completed


def _ray_asyncio_start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


class RayPoolExecutor(Executor, Parameters):
    """
    An executor that limits the number of concurrent Ray tasks by maintaining a pool of running tasks.
    Unlike ThreadPoolExecutor which pre-allocates threads, this executor dynamically manages Ray tasks
    using asyncio to control concurrency.

    Example usage:
        >>> executor = RayPoolExecutor(max_workers=4)  ## Must have ray installed
        >>> future = executor.submit(
                my_function, 
                arg1, 
                arg2,
                num_cpus=2  ## Allocate 2 CPUs for this task
            )
        >>> result = ray.get(future)  ## Wait for and retrieve the result

    Attributes:
        max_workers: Maximum number of concurrent Ray tasks. Pass float('inf') for unlimited tasks.
        iter_wait: Time to wait between iterations when checking task completion.
        item_wait: Time to wait between checking individual tasks.
    """

    max_workers: Union[int, Literal[inf]]
    iter_wait: float = _RAY_ACCUMULATE_ITER_WAIT
    item_wait: float = _RAY_ACCUMULATE_ITEM_WAIT
    _asyncio_event_loop: Optional = None
    _asyncio_event_loop_thread: Optional = None
    _submission_executor: Optional[ThreadPoolExecutor] = None
    _running_tasks: Dict[str, Any] = {}  ## Maps task_uid to Ray ObjectRef
    _latest_submit: Optional[int] = None

    def _set_asyncio(self):
        """
        Lazily initializes the asyncio event loop and its thread. This is done on-demand to avoid
        creating resources when the executor is not used with a worker limit.
        """
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
            fn: Callable,
            *args,
            scheduling_strategy: str = "SPREAD",
            num_cpus: int = 1,
            num_gpus: int = 0,
            max_retries: int = 0,
            retry_exceptions: Union[List, bool] = True,
            **kwargs,
    ):
        """
        Submits a function for execution using Ray. When max_workers is infinite, tasks are submitted 
        directly to Ray. Otherwise, uses asyncio to limit concurrent tasks.

        Args:
            fn: Function to execute as a task
            scheduling_strategy: Ray's scheduling strategy ("SPREAD" distributes tasks evenly across nodes).
            num_cpus: Number of CPUs required per task
            num_gpus: Number of GPUs required per task
            max_retries: Number of times to retry failed tasks
            retry_exceptions: Which exceptions should trigger retries
            *args, **kwargs: Arguments passed to fn

        Returns:
            If max_workers is inf: Ray ObjectRef
            Otherwise: asyncio.Future that resolves to a Ray ObjectRef
        """
        # print(f'Running {fn_str(fn)} using {Parallelize.ray} with num_cpus={num_cpus}, num_gpus={num_gpus}')
        _check_is_ray_installed()

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
        return fut

    async def _ray_run_fn_async(
            self,
            submit_task: Callable,
            task_uid: str,
    ):
        """
        Coroutine that manages task submission while respecting max_workers limit.
        Waits for task slots to become available by checking completion of existing tasks.

        Example of how tasks are managed:
            If max_workers=2 and 2 tasks are running:
            1. New task arrives, waits in while loop
            2. Loop checks existing tasks, finds completed task
            3. Completed task is removed, new task starts
            4. Process repeats for subsequent tasks

        Args:
            submit_task: Callback that creates and submits the Ray task
            task_uid: Unique identifier for tracking this task
        """
        # self._running_tasks[task_uid] = None
        ## Wait until we have capacity to run another task:
        while len(self._running_tasks) >= self.max_workers:
            ## Polling step: remove completed tasks until we have capacity:
            for _task_uid in sorted(self._running_tasks.keys()):
                if is_done(self._running_tasks[_task_uid]):
                    self._running_tasks.pop(_task_uid, None)  ## Task has completed, forget about it
                    if len(self._running_tasks) < self.max_workers:
                        break  ## Move onto next step to submit the task.
                time.sleep(self.item_wait)
            if len(self._running_tasks) < self.max_workers:
                break  ## Break the outer loop to submit the task.
            ## There is not enough capacity, keep waiting:
            time.sleep(self.iter_wait)  
            
        ## Now that we have capacity, submit the task and track it:
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
    _check_is_ray_installed()
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
@contextmanager
def RayDaskPersistWaitCallback():  ## Dummy contextmanager for cases when ray or dask is not installed.
    yield


if _IS_RAY_INSTALLED and _IS_DASK_INSTALLED:
    import ray
    from ray.util.dask import RayDaskCallback


    class RayDaskPersistWaitCallback(RayDaskCallback):
        ## Callback to wait for computation to complete when .persist() is called with block=True
        def _ray_postsubmit_all(self, object_refs, dsk):
            wait(object_refs)


def max_num_resource_actors(
        model_num_resources: Union[conint(ge=0), confloat(ge=0.0, lt=1.0)],
        ray_num_resources: int,
) -> Union[int, float]:
    ## Returns number of models possible, restricted by a particular resource; takes into account
    ## fractional resource requirements.
    ## Note: all resource-requirements are either 0, a float between 0 and 1, or an integer above 1.
    if model_num_resources == 0:
        return math.inf
    elif 0 < model_num_resources < 1:
        ## E.g. when a model needs <1 GPU, multiple models can occupy the same GPU.
        max_num_models_per_resource: int = math.floor(1 / model_num_resources)
        return ray_num_resources * max_num_models_per_resource
    else:
        ## E.g. when a model needs >1 GPU, it must be the only model occupying that GPU.
        return math.floor(ray_num_resources / model_num_resources)


class RayInitConfig(UserEnteredParameters):
    class Config(UserEnteredParameters.Config):
        extra = Extra.allow

    ## Default values:
    address: str = 'auto'
    temp_dir: Optional[str] = None
    include_dashboard: bool = False
    runtime_env: RayRuntimeEnv = {}


RayActorComposite = "RayActorComposite"


class RayActorComposite(Parameters):
    actor_id: str
    actor: Any
    request_counter: Any

    def kill(self):
        get_result(ray.kill(self.actor), wait=_RAY_ACCUMULATE_ITER_WAIT)
        get_result(ray.kill(self.request_counter), wait=_RAY_ACCUMULATE_ITER_WAIT)
        actor: ray.actor.ActorHandle = self.actor
        request_counter: ray.actor.ActorHandle = self.request_counter
        del actor
        del request_counter

    @classmethod
    def create_actors(
            cls,
            actor_factory: Callable,
            *,
            num_actors: int,
            request_counter_num_cpus: float = 0.1,
            request_counter_max_concurrency: int = 1000,
            **kwargs
    ) -> List[RayActorComposite]:
        progress_bar: Optional[Dict] = Alias.get_progress_bar(kwargs)
        actors_progress_bar: ProgressBar = ProgressBar.of(
            progress_bar,
            total=num_actors,
            desc=f'Creating Ray actors',
            unit='actors',
        )
        actor_ids: List[str] = as_list(String.random_name(num_actors))
        actor_composites: List[RayActorComposite] = []
        for actor_i, actor_id in zip(range(num_actors), actor_ids):
            request_counter: ray.actor.ActorHandle = RequestCounter.options(
                num_cpus=request_counter_num_cpus,
                max_concurrency=request_counter_max_concurrency,
            ).remote()
            actor: ray.actor.ActorHandle = actor_factory(
                request_counter=request_counter,
                actor_i=actor_i,
                actor_id=actor_id,
            )
            actor_composites.append(
                RayActorComposite(
                    actor_id=actor_id,
                    actor=actor,
                    request_counter=request_counter,
                )
            )
            actors_progress_bar.update(1)
            time.sleep(0.100)
        if len(actor_composites) != num_actors:
            msg: str = f'Creation of {num_actors - len(actor_composites)} actors failed'
            actors_progress_bar.failed(msg)
            raise ValueError(msg)
        else:
            msg: str = f'Created {num_actors} actors'
            actors_progress_bar.success(msg)
        return actor_composites
