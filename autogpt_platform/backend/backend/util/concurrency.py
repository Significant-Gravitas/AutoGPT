import asyncio
from collections.abc import Awaitable, Iterable
from typing import TypeVar

T = TypeVar("T")


async def gather_bounded(aws: Iterable[Awaitable[T]], limit: int) -> list[T]:
    """`asyncio.gather` that awaits at most `limit` of `aws` at a time.

    A coroutine in `aws` starts only once it has a slot, so a timeout it
    applies to itself does not count the wait for one.
    """
    slots = asyncio.Semaphore(limit)

    async def bounded(aw: Awaitable[T]) -> T:
        async with slots:
            return await aw

    return list(await asyncio.gather(*(bounded(aw) for aw in aws)))
