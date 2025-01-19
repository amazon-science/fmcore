"""A collection of concurrency utilities to augment the Python language:"""

import ctypes
import logging
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures._base import Future
from concurrent.futures.thread import BrokenThreadPool
from math import inf
from threading import Lock, Semaphore
from typing import *


class ThreadKilledSystemException(BaseException):
    """Custom exception for killing threads."""

    pass


class ThreadKilledSystemExceptionFilter(logging.Filter):
    def filter(self, record):
        if record.exc_info:
            exc_type = record.exc_info[0]
            if exc_type.__name__ == "ThreadKilledSystemException":
                return False
        return True


def suppress_ThreadKilledSystemException():
    for _logger_module in ["concurrent.futures", "ipykernel", "ipykernel.ipykernel"]:
        _logger = logging.getLogger(_logger_module)
        _filter_exists: bool = False
        for _filter in _logger.filters:
            if _filter.__class__.__name__ == "ThreadKilledSystemExceptionFilter":
                _filter_exists: bool = True
                # print(f'{_filter.__class__.__name__} exists in {_logger_module} filters')
                break
        if not _filter_exists:
            _logger.addFilter(ThreadKilledSystemExceptionFilter())
            # print(f'{ThreadKilledSystemExceptionFilter} added to {_logger_module} filters')


def kill_thread(tid: int):
    """
    Forces termination of a thread by injecting a ThreadKilledSystemException into it.
    This is a last-resort mechanism that should only be used when normal thread
    termination methods have failed.

    Technical Implementation:
        Uses the CPython C API (via ctypes) to inject an exception into the target thread's
        execution context. When the exception is raised, it will terminate the thread's
        execution at its next Python instruction.

    Example usage:
        >>> def long_running_task():
                while True:
                    time.sleep(1)  ## Simulate work

        >>> thread = threading.Thread(target=long_running_task)
        >>> thread.start()
        >>> thread_id = thread.ident
        >>> kill_thread(thread_id)  ## Thread will terminate on next instruction

    Intended usage:
        1. When performing concurrent/parallel tasks that may need to be cancelled after submission to a ThreadPoolExecutor:
            Example: Cancelling a task in an interactive Jupyter session:
            >>> prompt_template = "Who is the head of state in: {country}"
            >>> countries = ['USA', 'UK', 'India', 'China', 'Russia', ... ] ## Assume a large list
            >>> prompts = [prompt_template.format(country=country) for country in countries]
            >>> def call_llm(prompt) -> str:
                    return call_gpt(prompt)
            >>> ## Create a ThreadPoolExecutor:
                executor = ThreadPoolExecutor(max_workers=10)
            >>> ## Submit tasks to ThreadPoolExecutor:
                for gpt_generated_text in accumulate_iter([
                    run_concurrent(call_llm, prompt)
                    for prompt in prompt
                ]):  ## Waits for results as they complete and prints (may be out-of-order):
                    print(gpt_generated_text)
            >>> ## Now, suppose while printing the results, we realise the prompt is not good.
            >>> ## We want to cancel the pending tasks by pressing "stop" in Jupyter notebook.
            >>> ## By default, this will raise a KeyboardInterrupt, but WILL NOT stop the running tasks!
            >>> ## Instead, we can use kill_thread to stop the tasks:
            >>> executor.shutdown(wait=False)  ## Cancels pending items
            >>> for tid in worker_ids(executor):
                    kill_thread(tid)  ## After calling this, you can still submit
            >>> executor.shutdown(wait=False)  ## After calling this, you cannot submit

    Warning! Critical Thread-Safety expecations may be violated:
        1. Resource Cleanup:
            - Locks, file handles, and network connections may remain locked/open
            - Database transactions might be left uncommitted
            Example: If thread holds a lock when killed:
                >>> lock.acquire()
                >>> kill_thread(tid)  ## Lock remains acquired forever
        2. Data Integrity:
            - Shared data structures may be left in inconsistent states
            Example: During a multi-step update:
                >>> data['step1'] = new_value
                >>> kill_thread(tid)  ## 'step2' never happens, data is corrupt
        3. System Stability:
            - Python runtime isn't designed for forced thread termination
            - May cause memory leaks or interpreter instability
            Example: During critical system operations:
                >>> sys.modules['critical_module'] = new_module
                >>> kill_thread(tid)  ## System left in unknown state

    Args:
        tid: Thread ID (integer) of the thread to terminate. Obtain this from
             threading.Thread.ident

    Raises:
        ValueError: If tid is invalid
        TypeError: If exctype is not derived from BaseException
        SystemError: If thread termination fails
    """
    exctype: Type[BaseException] = ThreadKilledSystemException
    if not issubclass(exctype, BaseException):
        raise TypeError("Only types derived from BaseException are allowed")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    logging.debug(f"...killed thread ID: {tid}")
    if res == 0:
        raise ValueError(f"Invalid thread ID: {tid}")
    elif res != 1:
        # If it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def concurrent(max_workers: int = 10, max_calls_per_second: float = inf):
    """
    Decorator which runs function calls concurrently via multithreading.
    When decorating an IO-bound function with @concurrent(MAX_THREADS), and then invoking the function
    N times in a loop, it will run min(MAX_THREADS, N) invocations of the function concurrently.
    For example, if your function calls another service, and you must invoke the function N times, decorating with
    @concurrent(3) ensures that you only have 3 concurrent function-calls at a time, meaning you only make
    3 concurrent requests at a time. This reduces the number of connections you are making to the downstream service.
    As this uses multi-threading and not multi-processing, it is suitable for IO-heavy functions, not CPU-heavy.

    Each call  to the decorated function returns a future. Calling .result() on that future will return the value.
    Generally, you should call the decorated function N times in a loop, and store the futures in a list/dict. Then,
    call .result() on all the futures, saving the results in a new list/dict. Each .result() call is synchronous, so the
    order of items is maintained between the lists. When doing this, at most min(MAX_THREADS, N) function calls will be
    running concurrently.
    Note that if the function calls throws an exception, then calling .result() will raise the exception in the
    orchestrating code. If multiple function calls raise an exception, the one on which .result() was called first will
    throw the exception to the orchestrating code.  You should add try-catch logic inside your decorated function to
    ensure exceptions are handled.
    Note that decorated function `a` can call another decorated function `b` without issues; it is upto the function A
    to determine whether to call .result() on the futures it gets from `b`, or return the future to its own invoker.

    `max_calls_per_second` controls the rate at which we can call the function. This is particularly important for
    functions which execute quickly: e.g. suppose the decorated function calls a downstream service, and we allow a
    maximum concurrency of 5. If each function call takes 100ms, then we end up making 1000/100*5 = 50 calls to the
    downstream service each second. We thus should pass `max_calls_per_second` to restrict this to a smaller value.

    :param max_workers: the max number of threads which can be running the function at one time. This is thus
    them max concurrency factor.
    :param max_calls_per_second: controls the rate at which we can call the function.
    :return: N/A, this is a decorator.
    """

    ## Refs:
    ## 1. ThreadPoolExecutor: docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor.submit
    ## 2. Decorators: www.datacamp.com/community/tutorials/decorators-python
    ## 3. Semaphores: www.geeksforgeeks.org/synchronization-by-using-semaphore-in-python/
    ## 4. Overall code: https://gist.github.com/gregburek/1441055#gistcomment-1294264
    def decorator(function):
        ## Each decorated function gets its own executor and semaphore. These are defined at the function-level, so
        ## if you write two decorated functions `def say_hi` and `def say_bye`, they each gets a separate executor and
        ## semaphore. Then, if you invoke `say_hi` 30 times and `say_bye` 20 times, all 30 calls to say_hi will use the
        ## same executor and semaphore, and all 20 `say_bye` will use a different executor and semaphore. The value of
        ## `max_workers` will determine how many function calls actually run concurrently, e.g. if say_hi has
        ## max_workers=5, then the 30 calls will run 5 at a time (this is enforced by the semaphore).
        executor = ThreadPoolExecutor(max_workers=max_workers)
        semaphore = Semaphore(max_workers)

        ## The minimum time between invocations.
        min_time_interval_between_calls = 1 / max_calls_per_second
        ## This only stores a single value, but it must be a list (mutable) for Python's function scoping to work.
        time_last_called = [0.0]

        def wrapper(*args, **kwargs) -> Future:
            semaphore.acquire()
            time_elapsed_since_last_called = time.time() - time_last_called[0]
            time_to_wait_before_next_call = max(
                0.0, min_time_interval_between_calls - time_elapsed_since_last_called
            )
            time.sleep(time_to_wait_before_next_call)

            def run_function(*args, **kwargs):
                try:
                    result = function(*args, **kwargs)
                finally:
                    semaphore.release()  ## If the function call throws an exception, release the semaphore.
                return result

            time_last_called[0] = time.time()
            return executor.submit(run_function, *args, **kwargs)  ## return a future

        return wrapper

    return decorator


class RestrictedConcurrencyThreadPoolExecutor(ThreadPoolExecutor):
    """
    This executor restricts concurrency (max active threads) and, optionally, rate (max calls per second).
    It is similar in functionality to the @concurrent decorator, but implemented at the executor level.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        *args,
        max_calls_per_second: float = float("inf"),
        **kwargs,
    ):
        if max_workers is None:
            max_workers: int = min(32, (mp.cpu_count() or 1) + 4)
        if not isinstance(max_workers, int) or (max_workers < 1):
            raise ValueError("Expected `max_workers`to be a non-negative integer.")
        kwargs["max_workers"] = max_workers
        super().__init__(*args, **kwargs)
        self._semaphore = Semaphore(max_workers)
        self._max_calls_per_second = max_calls_per_second

        # If we have an infinite rate, don't enforce a delay
        self._min_time_interval_between_calls = 1 / self._max_calls_per_second

        # Tracks the last time a call was started (not finished, just started)
        self._time_last_called = 0.0
        self._lock = Lock()  # Protects access to _time_last_called

    def submit(self, fn, *args, **kwargs):
        # Enforce concurrency limit
        self._semaphore.acquire()

        # Rate limiting logic: Before starting a new call, ensure we wait long enough if needed
        if self._min_time_interval_between_calls > 0.0:
            with self._lock:
                time_elapsed_since_last_called = time.time() - self._time_last_called
                time_to_wait = max(
                    0.0, self._min_time_interval_between_calls - time_elapsed_since_last_called
                )

            # Wait the required time
            if time_to_wait > 0:
                time.sleep(time_to_wait)

            # Update the last-called time after the wait
            with self._lock:
                self._time_last_called = time.time()
        else:
            # No rate limiting, just update the last-called time
            with self._lock:
                self._time_last_called = time.time()

        future = super().submit(fn, *args, **kwargs)
        # When the task completes, release the semaphore to allow another task to start
        future.add_done_callback(lambda _: self._semaphore.release())
        return future


_GLOBAL_THREAD_POOL_EXECUTOR = None
_GLOBAL_THREAD_POOL_EXECUTOR_MAX_WORKERS: int = 16


def run_concurrent(
    fn,
    *args,
    executor: Optional[ThreadPoolExecutor] = None,
    **kwargs,
):
    global _GLOBAL_THREAD_POOL_EXECUTOR
    if _GLOBAL_THREAD_POOL_EXECUTOR is None:
        _GLOBAL_THREAD_POOL_EXECUTOR = RestrictedConcurrencyThreadPoolExecutor(
            max_workers=_GLOBAL_THREAD_POOL_EXECUTOR_MAX_WORKERS
        )
    if executor is None:
        executor: ThreadPoolExecutor = _GLOBAL_THREAD_POOL_EXECUTOR
    try:
        # logging.debug(f'Running {fn_str(fn)} using {Parallelize.threads} with max_workers={executor._max_workers}')
        return executor.submit(fn, *args, **kwargs)  ## return a future
    except BrokenThreadPool as e:
        if executor is _GLOBAL_THREAD_POOL_EXECUTOR:
            executor = RestrictedConcurrencyThreadPoolExecutor(
                max_workers=_GLOBAL_THREAD_POOL_EXECUTOR_MAX_WORKERS
            )
            del _GLOBAL_THREAD_POOL_EXECUTOR
            _GLOBAL_THREAD_POOL_EXECUTOR = executor
            return executor.submit(fn, *args, **kwargs)  ## return a future
        raise e


suppress_ThreadKilledSystemException()
