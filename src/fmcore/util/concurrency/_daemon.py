import logging
import time
import traceback
from datetime import datetime
from typing import *


def daemon(wait: float, exit_on_error: bool = False, sentinel: Optional[List] = None, **kwargs):
    """
    A decorator which runs a function as a daemon process in a background thread.

    You do not need to invoke this function directly: simply decorating the daemon function will start running it
    in the background.

    Example using class method: your daemon should be marked with @staticmethod. Example:
        class Printer:
            DATA_LIST = []
            @staticmethod
            @daemon(wait=3, mylist=DATA_LIST)
            def printer_daemon(mylist):
                if len(mylist) > 0:
                    print(f'Contents of list: {mylist}', flush=True)

    Example using sentinel:
        run_sentinel = [True]
        @daemon(wait=1, sentinel=run_sentinel)
        def run():
            print('Running', flush=True)
        time.sleep(3)  ## Prints "Running" 3 times.
        run_sentinel.pop()  ## Stops "Running" from printing any more.

    :param wait: the wait time in seconds between invocations to the @daemon decorated function.
    :param exit_on_error: whether to stop the daemon if an error is raised.
    :sentinel: can be used to stop the executor. When not passed, the daemon runs forever. When passed, `sentinel` must
        be a list with exactly one element (it can be anything). To stop the daemon, run "sentinel.pop()". It is
        important to pass a list (not a tuple), since lists are mutable, and thus the same exact object is used by
        both the executor and by the caller.
    :param kwargs: list of arguments passed to the decorator, which are forwarded to the decorated function as kwargs.
        These values will never change for the life of the daemon. However, if you pass references to mutables such as
        lists, dicts, objects etc to the decorator and use them in the daemon function, you can run certain tasks at a
        regular cadence on fresh data.
    :return: None
    """

    ## Refs on how decorators work:
    ## 1. https://www.datacamp.com/community/tutorials/decorators-python
    def decorator(function):
        ## Each decorated function gets its own executor. These are defined at the function-level, so
        ## if you write two decorated functions `def say_hi` and `def say_bye`, they each gets a separate
        ## executor. The executor for `say_hi` will call `say_hi` repeatedly, and the executor for `say_bye` will call
        ## `say_bye` repeatedly; they will not interact.
        executor = RestrictedConcurrencyThreadPoolExecutor(max_workers=1)

        def run_function_forever(sentinel):
            while sentinel is None or len(sentinel) > 0:
                start = time.perf_counter()
                try:
                    function(**kwargs)
                except Exception as e:
                    logging.debug(traceback.format_exc())
                    if exit_on_error:
                        raise e
                end = time.perf_counter()
                time_to_wait: float = max(0.0, wait - (end - start))
                time.sleep(time_to_wait)
            del executor  ## Cleans up the daemon after it finishes running.

        if sentinel is not None:
            if not isinstance(sentinel, list) or len(sentinel) != 1:
                raise ValueError("When passing `sentinel`, it must be a list with exactly one item.")
        executor.submit(run_function_forever, sentinel=sentinel)

        ## The wrapper here should do nothing, since you cannot call the daemon explicitly.
        def wrapper(*args, **kwargs):
            raise RuntimeError("Cannot call daemon function explicitly")

        return wrapper

    return decorator


## Dict of daemon ids to their sentinels
_DAEMONS: Dict[str, List[bool]] = {}


def start_daemon(
    fn,
    wait: float,
    daemon_id: Optional[str] = None,
    daemons: Dict[str, List[bool]] = _DAEMONS,
    **kwargs,
) -> str:
    assert isinstance(daemons, dict)
    assert isinstance(wait, (int, float)) and wait >= 0.0
    if daemon_id is None:
        dt: datetime = datetime.now()
        dt: datetime = dt.replace(tzinfo=dt.astimezone().tzinfo)
        if dt.tzinfo is not None:
            daemon_id: str = dt.strftime("%Y-%m-%d %H:%M:%S.%f UTC%z").strip()
        else:
            daemon_id: str = dt.strftime("%Y-%m-%d %H:%M:%S.%f").strip()
    assert isinstance(daemon_id, str) and len(daemon_id) > 0
    assert daemon_id not in daemons, f'Daemon with id "{daemon_id}" already exists.'

    daemon_sentinel: List[bool] = [True]

    @daemon(wait=wait, sentinel=daemon_sentinel)
    def run():
        fn(**kwargs)

    daemons[daemon_id] = daemon_sentinel
    return daemon_id


def stop_daemon(daemon_id: str, daemons: Dict[str, List[bool]] = _DAEMONS) -> bool:
    assert isinstance(daemons, dict)
    assert isinstance(daemon_id, str) and len(daemon_id) > 0
    daemon_sentinel: List[bool] = daemons.pop(daemon_id, [False])
    assert len(daemon_sentinel) == 1
    return daemon_sentinel.pop()
