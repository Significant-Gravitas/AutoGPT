import ast
import asyncio
from pathlib import Path

import pytest

from backend.voice_relay import waits

RELAY = Path(waits.__file__).parent


async def test_a_bounded_wait_answers_in_time_or_times_out():
    async def answer() -> str:
        await asyncio.sleep(0)
        return "answered"

    assert await waits.bounded(answer(), timeout=1) == "answered"
    assert await waits.bounded(answer(), timeout=None) == "answered"
    with pytest.raises(TimeoutError):
        await waits.bounded(asyncio.Event().wait(), timeout=0.01)


async def test_a_task_cancelled_as_its_bounded_wait_finishes_stays_cancelled():
    """The operation finishes in the step the waiting task is cancelled in,
    as a call's send does when the call ends: the task must still stop."""

    async def finishing_as_the_task_is_cancelled() -> str:
        await asyncio.sleep(0)
        task.cancel()
        return "sent"

    async def wait_then_carry_on() -> None:
        await waits.bounded(finishing_as_the_task_is_cancelled(), timeout=5)
        await asyncio.sleep(0)

    task = asyncio.create_task(wait_then_carry_on())
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_stopping_a_task_that_lost_its_cancellation_cancels_it_again():
    lost = []

    async def loses_its_first_cancellation() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            lost.append(True)
        await asyncio.Event().wait()

    task = asyncio.create_task(loses_its_first_cancellation())
    await asyncio.sleep(0)

    async with asyncio.timeout(5):
        await waits.stop([task], again_after_s=0.01)

    assert lost == [True]
    assert task.cancelled()


def test_the_relay_bounds_its_waits_through_bounded_alone():
    """``asyncio.wait_for`` can lose a cancellation on Python 3.11."""
    calls = []
    for path in sorted(RELAY.glob("*.py")):
        if path.name.endswith("_test.py") or path.name == "conftest.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr == "wait_for":
                calls.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.ImportFrom) and node.module == "asyncio":
                calls.extend(
                    f"{path.name}:{node.lineno}"
                    for alias in node.names
                    if alias.name == "wait_for"
                )
    assert calls == []
