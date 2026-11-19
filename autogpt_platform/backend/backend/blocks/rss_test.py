import asyncio
from datetime import datetime, timedelta, timezone
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
    monkeypatch.setattr("backend.blocks.rss.sleep", sleep)

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
    monkeypatch.setattr("backend.blocks.rss.sleep", sleep)

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        polling_rate=420,
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
    assert sleep.await_args_list == [call(420)] * 2


@pytest.mark.parametrize("polling_rate", [-1, 0, 1, 59])
async def test_continuous_mode_waits_at_least_the_minimum(monkeypatch, polling_rate):
    """A polling rate below the minimum waits the minimum between polls."""
    block = ReadRSSFeedBlock()
    sleep = AsyncMock(side_effect=asyncio.CancelledError)
    monkeypatch.setattr(block, "parse_feed", AsyncMock(return_value={"entries": []}))
    monkeypatch.setattr("backend.blocks.rss.sleep", sleep)

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        polling_rate=polling_rate,
        run_continuously=True,
    )
    with pytest.raises(asyncio.CancelledError):
        _ = [output async for output in block.run(input_data)]

    sleep.assert_awaited_once_with(ReadRSSFeedBlock.MIN_POLLING_RATE_SECONDS)


async def test_continuous_mode_outputs_each_entry_once(monkeypatch):
    """A later poll outputs only the entries newer than those already output."""
    block = ReadRSSFeedBlock()
    published = datetime.now(timezone.utc).replace(microsecond=0)
    old = _feed_entry("old", published)
    new = _feed_entry("new", published + timedelta(minutes=1))
    parse_feed = AsyncMock(
        side_effect=[{"entries": [old]}, {"entries": [old]}, {"entries": [new, old]}]
    )
    sleep = AsyncMock(side_effect=[None, None, asyncio.CancelledError])
    monkeypatch.setattr(block, "parse_feed", parse_feed)
    monkeypatch.setattr("backend.blocks.rss.sleep", sleep)

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        polling_rate=60,
        run_continuously=True,
    )
    outputs = []
    with pytest.raises(asyncio.CancelledError):
        async for name, value in block.run(input_data):
            outputs.append(
                (name, [e.title for e in value] if name == "entries" else value.title)
            )

    assert outputs == [
        ("entry", "old"),
        ("entries", ["old"]),
        ("entries", []),
        ("entry", "new"),
        ("entries", ["new"]),
    ]


def _feed_entry(title: str, published: datetime) -> dict:
    return {
        "title": title,
        "link": f"https://example.com/{title}",
        "published_parsed": published.timetuple(),
    }
