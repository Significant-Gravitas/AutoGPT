import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, call

import pytest

from backend.blocks.rss import ReadRSSFeedBlock, RSSEntry


@pytest.mark.parametrize("has_entries", [False, True])
@pytest.mark.parametrize("polling_rate", [0, 60, 300])
async def test_run_once_does_not_sleep(monkeypatch, has_entries, polling_rate):
    """Run once emits its results and finishes without a polling wait."""
    block = ReadRSSFeedBlock()
    published = datetime.now(timezone.utc).replace(microsecond=0)
    entry = {
        "title": "Example RSS item",
        "link": "https://example.com/article",
        "published_parsed": published.timetuple(),
    }
    parse_feed = AsyncMock(return_value={"entries": [entry] if has_entries else []})
    sleep = AsyncMock()
    monkeypatch.setattr(block, "parse_feed", parse_feed)
    monkeypatch.setattr("backend.blocks.rss.asyncio.sleep", sleep)

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        polling_rate=polling_rate,
        run_continuously=False,
    )
    outputs = [output async for output in block.run(input_data)]

    expected_entry = RSSEntry(
        title=entry["title"],
        link=entry["link"],
        description="",
        pub_date=published,
        author="",
        categories=[],
    )
    assert outputs == (
        [("entry", expected_entry), ("entries", [expected_entry])]
        if has_entries
        else [("entries", [])]
    )
    parse_feed.assert_awaited_once_with(input_data.rss_url)
    sleep.assert_not_called()


async def test_continuous_mode_sleeps_between_polls(monkeypatch):
    """Continuous mode waits between polls and remains cancellable."""
    block = ReadRSSFeedBlock()
    parse_feed = AsyncMock(side_effect=[{"entries": []}, {"entries": []}])
    sleep = AsyncMock(side_effect=[None, asyncio.CancelledError])
    monkeypatch.setattr(block, "parse_feed", parse_feed)
    monkeypatch.setattr("backend.blocks.rss.asyncio.sleep", sleep)

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        polling_rate=42,
        run_continuously=True,
    )

    outputs = []
    with pytest.raises(asyncio.CancelledError):
        async for output in block.run(input_data):
            outputs.append(output)
            assert parse_feed.await_count == len(outputs)
            assert sleep.await_count == len(outputs) - 1

    assert outputs == [("entries", []), ("entries", [])]
    assert parse_feed.await_args_list == [call(input_data.rss_url)] * 2
    assert sleep.await_args_list == [call(42)] * 2
