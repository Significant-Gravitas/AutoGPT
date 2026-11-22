"""The look for cua-driver, when it takes longer than its callers wait.

A start waits RESTART_LOOKUP_SECONDS for the look and then fails; HELLO, the
preflight and the display watch wait their own few seconds. With a look that
takes longer than that every time (a slow share on PATH, or among the folders
Python imports from), each caller gave up, the look found the driver a moment
later with nobody waiting, and the next caller began another look that took
as long. The driver was there and never started, however often it was tried.

The answer of a look that outlasted a caller is kept now, for
LATE_ANSWER_SECONDS: the callers that come in that time take it without
another look, and the first start among them uses it up, so that the start
after it looks again and finds a driver that was moved or upgraded.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from autogpt_local_executor.computer_use import driver
from autogpt_local_executor.computer_use.driver import CuaDriverBridge, DriverUnavailable
from autogpt_local_executor.protocol import ErrorCode
from tests.cua_fakes import FAKE, fake_bridge, fake_log

__all__ = ["fake_log"]  # a fixture, used by name

# Longer than any wait here, shorter than a run that went wrong should take.
WITHIN = 60


class _SlowLook:
    """A look for the driver that takes until `answers` is set, each time."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, found: Path | None) -> None:
        self.answers = threading.Event()
        self.found = found
        self.looks = 0
        monkeypatch.setattr(driver, "find_driver_binary", self)

    def __call__(self, explicit: Any = None) -> Path | None:
        self.looks += 1
        self.answers.wait(WITHIN)
        return self.found


@pytest.fixture
def look(monkeypatch: pytest.MonkeyPatch, fake_log: Path) -> _SlowLook:
    monkeypatch.setattr(driver, "RESTART_LOOKUP_SECONDS", 0.2)
    return _SlowLook(monkeypatch, FAKE.absolute())


async def _answer_with_nobody_waiting(look: _SlowLook) -> None:
    """Let the look under way answer, and make the next one as slow."""
    look.answers.set()
    async with asyncio.timeout(WITHIN):
        while _looking():
            await asyncio.sleep(0.01)
    # The answer reaches the event loop a turn after its thread ends.
    await asyncio.sleep(0.05)
    look.answers.clear()


def _looking() -> bool:
    return any(thread.name == "autogpt-shim-driver-lookup" for thread in threading.enumerate())


async def _a_start_gives_up(bridge: CuaDriverBridge) -> None:
    with pytest.raises(DriverUnavailable, match="Looking for cua-driver"):
        await asyncio.wait_for(bridge.snapshot(), timeout=WITHIN)


async def test_a_start_takes_the_answer_that_came_after_the_last_start_gave_up(
    look: _SlowLook,
) -> None:
    bridge = fake_bridge()
    try:
        await _a_start_gives_up(bridge)
        await _answer_with_nobody_waiting(look)

        # Another look would take as long, and this start would give up too.
        snapshot = await asyncio.wait_for(bridge.snapshot(), timeout=WITHIN)

        assert snapshot.tools
        assert look.looks == 1
    finally:
        look.answers.set()
        await asyncio.wait_for(bridge.close(), timeout=WITHIN)


async def test_the_start_after_that_one_looks_again(look: _SlowLook) -> None:
    """The kept answer is one start's: a driver that was moved or upgraded
    since is found by the next."""
    bridge = fake_bridge()
    try:
        await _a_start_gives_up(bridge)
        await _answer_with_nobody_waiting(look)
        await asyncio.wait_for(bridge.snapshot(), timeout=WITHIN)
        assert look.looks == 1
        # The driver exits, and is started again.
        with pytest.raises(DriverUnavailable, match="stopped during the call"):
            await bridge.call_tool("drag", {"pid": -3}, timeout=30)
        look.answers.set()

        await asyncio.wait_for(bridge.snapshot(), timeout=WITHIN)

        assert look.looks == 2
    finally:
        look.answers.set()
        await asyncio.wait_for(bridge.close(), timeout=WITHIN)


async def test_an_answer_kept_too_long_is_not_used(look: _SlowLook, monkeypatch) -> None:
    monkeypatch.setattr(driver, "LATE_ANSWER_SECONDS", 0.0, raising=False)
    bridge = fake_bridge()
    try:
        await _a_start_gives_up(bridge)
        await _answer_with_nobody_waiting(look)
        look.answers.set()

        await asyncio.wait_for(bridge.snapshot(), timeout=WITHIN)

        assert look.looks == 2
    finally:
        look.answers.set()
        await asyncio.wait_for(bridge.close(), timeout=WITHIN)


async def test_hello_takes_the_kept_answer_and_leaves_it_for_the_start(look: _SlowLook) -> None:
    """HELLO, the preflight and the display watch ask whether the driver is
    there, and start nothing with the answer: the start that follows a HELLO
    which offered computer use has to find the answer still kept."""
    bridge = fake_bridge()
    try:
        # A HELLO gives up on the look, as a start does.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(bridge.find_binary(), timeout=0.2)
        await _answer_with_nobody_waiting(look)

        for _ in range(2):
            found = await asyncio.wait_for(bridge.find_binary(), timeout=5)
            assert found == FAKE.absolute()
        snapshot = await asyncio.wait_for(bridge.snapshot(), timeout=WITHIN)

        assert snapshot.tools
        assert look.looks == 1
    finally:
        look.answers.set()
        await asyncio.wait_for(bridge.close(), timeout=WITHIN)


async def test_a_look_that_found_nothing_too_late_is_reported_as_not_installed(
    look: _SlowLook,
) -> None:
    """Otherwise every start said the look took too long, and none said
    what it found: that the driver isn't installed, and how to add it."""
    look.found = None
    bridge = fake_bridge()
    try:
        await _a_start_gives_up(bridge)
        await _answer_with_nobody_waiting(look)

        with pytest.raises(DriverUnavailable, match="not installed") as missing:
            await asyncio.wait_for(bridge.snapshot(), timeout=WITHIN)

        assert missing.value.code == ErrorCode.DEPENDENCY_MISSING
        assert driver.COMPUTER_USE_INSTALL in missing.value.message
        assert look.looks == 1
    finally:
        look.answers.set()
        await asyncio.wait_for(bridge.close(), timeout=WITHIN)


async def test_an_answer_its_caller_waited_for_is_not_kept(
    monkeypatch: pytest.MonkeyPatch, fake_log: Path
) -> None:
    """A look that answers within its callers' waits, the usual one, is as
    before: each start looks afresh."""
    look = _SlowLook(monkeypatch, FAKE.absolute())
    look.answers.set()
    bridge = fake_bridge()
    try:
        assert await bridge.find_binary() == FAKE.absolute()
        assert await bridge.find_binary() == FAKE.absolute()
        assert look.looks == 2

        await asyncio.wait_for(bridge.snapshot(), timeout=WITHIN)
        with pytest.raises(DriverUnavailable, match="stopped during the call"):
            await bridge.call_tool("drag", {"pid": -3}, timeout=30)
        await asyncio.wait_for(bridge.snapshot(), timeout=WITHIN)

        assert look.looks == 4
    finally:
        await asyncio.wait_for(bridge.close(), timeout=WITHIN)
