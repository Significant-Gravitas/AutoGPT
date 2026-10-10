"""Tests for ReadRSSFeedBlock polling behaviour.

Regression coverage for the run-once case: the block must not wait out its
polling interval when it is only going to run a single time.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.blocks.rss import ReadRSSFeedBlock


class _StopLoop(Exception):
    """Raised by the sleep mock to break the continuous loop after one poll."""


async def _drain(agen) -> None:
    async for _ in agen:
        pass


async def test_run_once_does_not_sleep(monkeypatch):
    """run_continuously=False must return without waiting the polling interval."""
    block = ReadRSSFeedBlock()
    sleep = AsyncMock()
    monkeypatch.setattr(block, "parse_feed", AsyncMock(return_value={"entries": []}))
    monkeypatch.setattr(asyncio, "sleep", sleep)

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        polling_rate=60,
        run_continuously=False,
    )
    await _drain(block.run(input_data))

    sleep.assert_not_awaited()


async def test_continuous_mode_sleeps_between_polls(monkeypatch):
    """run_continuously=True must still wait the polling interval between polls."""
    block = ReadRSSFeedBlock()
    # Break the (otherwise infinite) loop after the first wait is observed.
    sleep = AsyncMock(side_effect=_StopLoop)
    monkeypatch.setattr(block, "parse_feed", AsyncMock(return_value={"entries": []}))
    monkeypatch.setattr(asyncio, "sleep", sleep)

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        polling_rate=42,
        run_continuously=True,
    )

    with pytest.raises(_StopLoop):
        await _drain(block.run(input_data))

    sleep.assert_awaited_once_with(42)
