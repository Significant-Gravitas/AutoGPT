"""Run synchronous cache invalidations to completion after their caller cancels."""

import asyncio
import contextvars
import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import ParamSpec

logger = logging.getLogger(__name__)
P = ParamSpec("P")
_cache_executor = ThreadPoolExecutor(thread_name_prefix="cache-invalidation")


async def run_cache_invalidation(
    invalidate: Callable[P, object], /, *args: P.args, **kwargs: P.kwargs
) -> None:
    await run_cache_invalidation_in(_cache_executor, invalidate, *args, **kwargs)


async def run_cache_invalidation_in(
    executor: ThreadPoolExecutor,
    invalidate: Callable[P, object],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    context = contextvars.copy_context()
    worker = executor.submit(context.run, invalidate, *args, **kwargs)
    result = asyncio.wrap_future(worker)
    try:
        await asyncio.shield(result)
    except asyncio.CancelledError:
        # Observe the asyncio future if the loop survives. Log from the worker
        # future so a failure is still reported after the request loop closes.
        result.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        worker.add_done_callback(lambda done: context.run(_log_failure, done))
        raise


def _log_failure(worker: Future[object]) -> None:
    try:
        worker.result()
    except Exception:
        logger.exception("Cache invalidation failed after request cancellation")
