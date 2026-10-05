"""Each compaction's callbacks share one rebuild and answer in part order."""

import asyncio

import pytest

from .after_compaction_handout import AfterCompactionHandout


def _rebuilds(*versions: list[str]):
    """A rebuild that returns ``versions`` one call at a time, and its calls."""
    remaining = iter(versions)
    calls: list[int] = []

    async def rebuild() -> list[str]:
        calls.append(1)
        await asyncio.sleep(0)
        return next(remaining)

    return rebuild, calls


@pytest.mark.asyncio
async def test_any_callback_can_open_the_round():
    rebuild, calls = _rebuilds(["one", "two"])
    handout = AfterCompactionHandout(rebuild, max_parts=3)

    last, middle, first = await asyncio.gather(
        handout.part(2), handout.part(1), handout.part(0)
    )

    assert (first, middle, last) == ("one", "two", "")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_part_asked_for_again_opens_the_next_round():
    rebuild, calls = _rebuilds(["a1", "a2"], ["b1", "b2"])
    handout = AfterCompactionHandout(rebuild, max_parts=2)

    first = await asyncio.gather(handout.part(0), handout.part(1))
    second = await asyncio.gather(handout.part(1), handout.part(0))

    assert first == ["a1", "a2"]
    assert second == ["b2", "b1"]
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_a_cancelled_callback_leaves_the_rebuild_for_the_others():
    release = asyncio.Event()

    async def rebuild() -> list[str]:
        await release.wait()
        return ["one", "two"]

    handout = AfterCompactionHandout(rebuild, max_parts=2)
    first = asyncio.create_task(handout.part(0))
    second = asyncio.create_task(handout.part(1))
    await asyncio.sleep(0)

    first.cancel()
    release.set()

    assert await second == "two"
    with pytest.raises(asyncio.CancelledError):
        await first


@pytest.mark.asyncio
async def test_a_part_does_not_wait_forever_for_one_that_never_comes():
    rebuild, _ = _rebuilds(["one", "two"])
    handout = AfterCompactionHandout(rebuild, max_parts=2, order_wait_s=0.01)

    assert await asyncio.wait_for(handout.part(1), timeout=5) == "two"
