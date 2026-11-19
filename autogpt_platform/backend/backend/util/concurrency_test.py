import asyncio

import pytest

from backend.util.concurrency import gather_bounded


@pytest.mark.asyncio
async def test_a_coroutine_starts_only_once_it_has_a_slot():
    release = asyncio.Event()
    started: list[int] = []

    async def work(n: int) -> int:
        started.append(n)
        await release.wait()
        return n

    gathering = asyncio.create_task(gather_bounded([work(n) for n in range(5)], 2))
    for _ in range(10):
        await asyncio.sleep(0)

    assert started == [0, 1]
    release.set()
    assert await asyncio.wait_for(gathering, timeout=10) == [0, 1, 2, 3, 4]
    assert started == [0, 1, 2, 3, 4]
