import multiprocessing as mp
import queue
import random
import threading
import traceback
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures._base import Executor, Future
from concurrent.futures.process import BrokenProcessPool
from typing import *

import cloudpickle

from fmcore.constants.DataProcessingConstants import Status

from ._utils import LoadBalancingStrategy


def actor_process_main(cls_bytes, init_args, init_kwargs, command_queue, result_queue):
    cls = cloudpickle.loads(cls_bytes)
    instance = None
    while True:
        command = command_queue.get()
        if command is None:
            break
        request_id, method_name, args, kwargs = command
        try:
            if method_name == "__initialize__":
                instance = cls(*init_args, **init_kwargs)
                result_queue.put((request_id, "ok", None))
                continue
            if instance is None:
                raise RuntimeError("Actor instance not initialized.")
            method = getattr(instance, method_name, None)
            if method is None:
                raise AttributeError(f"Method '{method_name}' not found.")
            result = method(*args, **kwargs)
            result_queue.put((request_id, "ok", result))
        except Exception as e:
            tb_str = traceback.format_exc()
            result_queue.put((request_id, "error", (e, tb_str)))


class ActorProxy:
    def __init__(
        self, cls, init_args, init_kwargs, mp_context: Literal["fork", "spawn"]
    ):
        assert mp_context in {"fork", "spawn"}
        ctx = mp.get_context(mp_context)

        self._uuid = str(uuid.uuid4())

        self._command_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._num_submitted: int = 0
        self._task_status: Dict[Status, int] = {
            Status.PENDING: 0,
            Status.RUNNING: 0,
            Status.SUCCEEDED: 0,
            Status.FAILED: 0,
        }

        self._futures = {}
        self._futures_lock = threading.Lock()

        # Create the process using the fork context
        cls_bytes = cloudpickle.dumps(cls)
        self._cls_name = cls.__name__
        self._process: ctx.Process = ctx.Process(
            target=actor_process_main,
            args=(
                cls_bytes,
                init_args,
                init_kwargs,
                self._command_queue,
                self._result_queue,
            ),
        )
        self._process.start()

        # Synchronous initialization
        self._invoke_sync_initialize()

        self._stopped = False

        # Now start the asynchronous result handling using a thread:
        self._result_thread = threading.Thread(target=self._handle_results, daemon=True)
        self._result_thread.start()

    def _handle_results(self):
        while True:
            if not self._process.is_alive() and self._result_queue.empty():
                self._task_status[Status.RUNNING] = 0
                return
            try:
                item = self._result_queue.get(timeout=1)
            except queue.Empty:
                self._task_status[Status.RUNNING] = 0
                continue
            if item is None:  # Sentinel to stop the results-handling thread.
                return
            request_id, status, payload = item
            with self._futures_lock:
                future = self._futures.pop(request_id, None)
            if future is not None:
                if status == "ok":
                    future.set_result(payload)
                    self._task_status[Status.SUCCEEDED] += 1
                else:
                    e, tb_str = payload
                    future.set_exception(RuntimeError(f"Remote call failed:\n{tb_str}"))
                    self._task_status[Status.FAILED] += 1
                self._task_status[Status.PENDING] -= 1

    def _invoke_sync_initialize(self):
        request_id = self._uuid
        self._command_queue.put((request_id, "__initialize__", (), {}))
        # Direct, blocking call to get the response
        rid, status, payload = self._result_queue.get()
        if status == "error":
            e, tb_str = payload
            raise RuntimeError(f"Remote init failed:\n{tb_str}")

    def stop(self, timeout: int = 10, cancel_futures: bool = True):
        if self._stopped is True:
            return
        self._stopped = True
        self._command_queue.put(None)
        self._process.join(timeout=timeout)
        self._command_queue.close()
        self._result_queue.close()
        # Fail any remaining futures
        if cancel_futures:
            with self._futures_lock:
                for fut in self._futures.values():
                    if not fut.done():
                        fut.set_exception(
                            RuntimeError("Actor stopped before completion.")
                        )
                self._futures.clear()
        self._task_status[Status.RUNNING] = 0

    def _invoke(self, method_name, *args, **kwargs):
        if self._stopped is True:
            raise RuntimeError("Cannot invoke methods on a stopped actor.")
        future = Future()
        request_id = str(uuid.uuid4())
        with self._futures_lock:
            self._futures[request_id] = future
        self._command_queue.put((request_id, method_name, args, kwargs))
        self._num_submitted += 1
        self._task_status[Status.PENDING] += 1
        if self._process.is_alive():
            self._task_status[Status.RUNNING] = 1
        return future

    def submitted(self) -> int:
        return self._num_submitted

    def pending(self) -> int:
        return self._task_status[Status.PENDING]

    def running(self) -> int:
        return self._task_status[Status.RUNNING]

    def succeeded(self) -> int:
        return self._task_status[Status.SUCCEEDED]

    def failed(self) -> int:
        return self._task_status[Status.FAILED]

    def __getattr__(self, name):
        # Instead of returning a direct callable, we return a RemoteMethod wrapper
        return RemoteMethod(self, name, self._cls_name)

    def __del__(self):
        try:
            if not self._stopped and self._process.is_alive():
                self.stop()
        except Exception:
            pass


class RemoteMethod:
    """
    A wrapper object returned by ActorProxy.__getattr__.
    To call the method remotely, use .remote(*args, **kwargs).
    """

    def __init__(self, proxy, method_name, cls_name):
        self._proxy = proxy
        self._method_name = method_name
        self._cls_name = cls_name

    def remote(self, *args, **kwargs):
        return self._proxy._invoke(self._method_name, *args, **kwargs)

    def options(self, *args, **kwargs):
        warnings.warn(
            f'The process-based Actor "{self._cls_name}" cannot use .options(); this call will be ignored.'
        )
        return self


"""
Note: By default we use a `mp_context="fork"` for Actor creation.
Process creation is much slower under spawn than forking. For example:
- On a MacOS machine, Actor creation time is 20 milliseconds (forking) vs 7 seconds (spawn).
- On a Linux machine, Actor creation time is 20 milliseconds (forking) vs 17 seconds (spawn).

However, forking comes with caveats which are not present in spawn:
1. Copy-on-Write Memory Behavior:
On Unix-like systems (including MacOS), forked processes share the same memory pages as the parent initially.
These pages are not immediately copied; instead, they are marked copy-on-write.
This means:
- No immediate bulk copy: Your large data structures (like Pandas DataFrames) do not get physically copied into memory
right away.
- Copies on modification: If either the parent or the child modifies a shared page, only then is that page actually
copied. Thus, if the child process reads from large data structures without writing to them, the overhead remains
relatively low. But if it modifies them, the memory cost could jump significantly.

2. Potential Resource and Concurrency Issues:
Forking a process that already has multiple threads, open file descriptors, or other system resources can lead to
subtle bugs. Some libraries, particularly those relying on threading or certain system calls, may not be “fork-safe.”
Common issues include:
- Thread State: The child process starts with a copy of the parent’s memory but only one thread running (the one that
called fork). Any locks or conditions held by threads in the parent at the time of fork can lead to deadlocks or
inconsistent states.
- External Resources: Network sockets, open database connections, or other system resources may not be safe to use in
the child after fork without an exec. They might appear duplicated but can behave unexpectedly or lead to errors if
not reinitialized.
- Library Incompatibilities: Some libraries are not tested or guaranteed to work correctly in forked children. They
might rely on internal threading, which can break post-fork.
"""
_DEFAULT_ACTOR_PROCESS_CREATION_METHOD: Literal["fork", "spawn"] = "fork"


class Actor:
    @classmethod
    def remote(
        cls,
        *args,
        mp_context: Literal["fork", "spawn"] = _DEFAULT_ACTOR_PROCESS_CREATION_METHOD,
        **kwargs,
    ):
        return ActorProxy(
            cls,
            init_args=args,
            init_kwargs=kwargs,
            mp_context=mp_context,
        )

    @classmethod
    def options(cls, *args, **kwargs):
        warnings.warn(
            f'The process-based Actor "{cls.__name__}" cannot use .options(); this call will be ignored.'
        )
        return cls


def actor(
    cls, mp_context: Literal["fork", "spawn"] = _DEFAULT_ACTOR_PROCESS_CREATION_METHOD
):
    """
    Class decorator that transforms a regular class into an actor-enabled class.
    The decorated class gains a .remote(*args, **kwargs) class method that
    returns an ActorProxy running in a separate process.
    """

    def remote(*args, **kwargs):
        return ActorProxy(
            cls,
            init_args=args,
            init_kwargs=kwargs,
            mp_context=mp_context,
        )

    def options(cls, *args, **kwargs):
        warnings.warn(
            f'The process-based Actor "{cls.__name__}" cannot use .options(); this call will be ignored.'
        )
        return cls

    cls.remote = remote
    cls.options = options
    return cls


@actor
class TaskActor:
    """
    A generic actor that can run an arbitrary callable passed to it.
    We'll send (func, args, kwargs) as serialized objects and it will run them.
    """

    def __init__(self):
        pass

    def run_callable(self, func_bytes, args, kwargs):
        func = cloudpickle.loads(func_bytes)
        return func(*args, **kwargs)


class ActorPoolExecutor(Executor):
    """
    A simple ActorPoolExecutor that mimics the ProcessPoolExecutor interface,
    but uses a pool of TaskActor instances for parallel execution.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        *,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
    ):
        if max_workers is None:
            max_workers = mp.cpu_count() - 1
        self._actors: List[ActorProxy] = [
            TaskActor.remote() for _ in range(max_workers)
        ]
        self._actor_index = 0
        self._max_workers = max_workers
        self._load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy(
            load_balancing_strategy
        )
        self._shutdown_lock = threading.Lock()
        self._futures = []
        self._shutdown = False

    def submit(self, fn, *args, **kwargs):
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("Cannot submit tasks after shutdown")

        func_bytes = cloudpickle.dumps(fn)
        if self._load_balancing_strategy is LoadBalancingStrategy.ROUND_ROBIN:
            actor = self._actors[self._actor_index]
            self._actor_index = (self._actor_index + 1) % self._max_workers
        elif self._load_balancing_strategy is LoadBalancingStrategy.RANDOM:
            actor = random.choice(self._actors)
        elif self._load_balancing_strategy is LoadBalancingStrategy.LEAST_USED:
            actor = sorted(
                [(_actor, _actor.pending()) for _actor in self._actors],
                key=lambda x: x[1],
            )[0]
        elif self._load_balancing_strategy is LoadBalancingStrategy.UNUSED:
            actor = sorted(
                [(_actor, _actor.running()) for _actor in self._actors],
                key=lambda x: x[1],
            )[0]
        else:
            raise NotImplementedError(
                f"Unsupported load_balancing_strategy: {self._load_balancing_strategy}"
            )
        future = actor.run_callable.remote(func_bytes, args, kwargs)
        self._remove_completed_futures()
        self._futures.append(future)
        return future

    def _remove_completed_futures(self):
        self._futures = [fut for fut in self._futures if not fut.done()]

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = True) -> None:
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True

        # If wait=True, wait for all futures to complete
        if wait:
            for fut in self._futures:
                fut.result()  # blocks until future is done or raises
        self._remove_completed_futures()
        # Stop all actors
        for actor in self._actors:
            actor.stop(cancel_futures=cancel_futures)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        if chunksize != 1:
            raise NotImplementedError("chunksize other than 1 is not implemented")

        inputs = zip(*iterables)
        futures = [self.submit(fn, *args) for args in inputs]

        # Yield results in order
        for fut in futures:
            yield fut.result(timeout=timeout)


_GLOBAL_PROCESS_POOL_EXECUTOR = None
_GLOBAL_PROCESS_POOL_EXECUTOR_MAX_WORKERS: int = max(1, min(32, mp.cpu_count() - 1))


def run_parallel(
    fn,
    *args,
    executor: Optional[Union[ProcessPoolExecutor, ActorPoolExecutor]] = None,
    **kwargs,
):
    global _GLOBAL_PROCESS_POOL_EXECUTOR
    if _GLOBAL_PROCESS_POOL_EXECUTOR is None:
        _GLOBAL_PROCESS_POOL_EXECUTOR = ActorPoolExecutor(
            max_workers=_GLOBAL_PROCESS_POOL_EXECUTOR_MAX_WORKERS
        )
    if executor is None:
        executor: ActorPoolExecutor = _GLOBAL_PROCESS_POOL_EXECUTOR
    try:
        # print(f'Running {fn_str(fn)} using {Parallelize.threads} with max_workers={executor._max_workers}')
        return executor.submit(fn, *args, **kwargs)  ## return a future
    except BrokenProcessPool as e:
        if executor is _GLOBAL_PROCESS_POOL_EXECUTOR:
            executor = ActorPoolExecutor(
                max_workers=_GLOBAL_PROCESS_POOL_EXECUTOR_MAX_WORKERS
            )
            del _GLOBAL_PROCESS_POOL_EXECUTOR
            _GLOBAL_PROCESS_POOL_EXECUTOR = executor
            return executor.submit(fn, *args, **kwargs)  ## return a future
        raise e
