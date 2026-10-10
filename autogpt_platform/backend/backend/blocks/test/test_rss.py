import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.blocks.rss import ReadRSSFeedBlock

FEED = {
    "entries": [
        {
            "title": "Example RSS Item",
            "link": "https://example.com/article",
            "summary": "This is an example RSS item description.",
            "published_parsed": (2023, 6, 23, 12, 30, 0, 4, 174, 0),
            "author": "John Doe",
            "tags": [{"term": "Technology"}],
        }
    ]
}


def _input(block: ReadRSSFeedBlock, run_continuously: bool):
    return block.Input(
        rss_url="https://example.com/rss",
        time_period=10_000_000,
        polling_rate=60,
        run_continuously=run_continuously,
    )


@pytest.mark.asyncio
async def test_run_once_does_not_sleep_for_polling_rate(monkeypatch):
    block = ReadRSSFeedBlock()
    sleep = AsyncMock()
    monkeypatch.setattr(block, "parse_feed", AsyncMock(return_value=FEED))
    monkeypatch.setattr(asyncio, "sleep", sleep)

    outputs = [out async for out in block.run(_input(block, False))]

    assert [name for name, _ in outputs] == ["entry", "entries"]
    sleep.assert_not_called()


@pytest.mark.asyncio
async def test_run_continuously_sleeps_for_polling_rate(monkeypatch):
    block = ReadRSSFeedBlock()
    parse_feed = AsyncMock(return_value=FEED)
    sleep = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(block, "parse_feed", parse_feed)
    monkeypatch.setattr(asyncio, "sleep", sleep)

    with pytest.raises(asyncio.CancelledError):
        async for _ in block.run(_input(block, True)):
            pass

    parse_feed.assert_awaited_once()
    sleep.assert_awaited_once_with(60)
