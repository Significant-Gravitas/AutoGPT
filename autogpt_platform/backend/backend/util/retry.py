import asyncio
import logging
import os
from typing import Any, Callable, Coroutine, TypeVar

from tenacity import retry, stop_after_attempt, wait_exponential

conn_retry = retry(
    stop=stop_after_attempt(30),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)
pid = os.getpid()


async def logged_retry(
    func: Callable[[], Coroutine[Any, Any, T]],
    resource_name: str,
    action_name: str,
    call_count: int = 0,
    max_retry: int = 5,
) -> T:
    prefix = f"[PID-{pid}|{resource_name}]"
    try:
        logger.info(f"{prefix} {action_name} started..")
        res = await func()
        logger.info(f"{prefix} {action_name} completed!")
        return res
    except Exception as e:
        if call_count <= max_retry:
            logger.info(f"{prefix} {action_name} failed: {e}. Retrying now..")
            await asyncio.sleep(2**call_count)
            return await logged_retry(
                func=func,
                resource_name=resource_name,
                action_name=action_name,
                call_count=call_count + 1,
                max_retry=max_retry,
            )
        else:
            raise e
