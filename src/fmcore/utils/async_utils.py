import asyncio
from concurrent.futures import ThreadPoolExecutor

import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncUtils:
    _executor = ThreadPoolExecutor()

    @staticmethod
    def execute(coro):
        """
        Executes an async coroutine in a thread pool executor.
        - No event loop interaction: Just submit coroutines to the thread pool.
        - Always uses ThreadPoolExecutor to run async functions.
        """
        return AsyncUtils._executor.submit(lambda: asyncio.run(coro)).result()

