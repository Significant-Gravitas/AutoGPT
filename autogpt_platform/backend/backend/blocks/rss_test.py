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


@pytest.mark.parametrize(
    "polls, expected",
    [
        ([["old"], ["old"], ["new", "old"]], [["old"], [], ["new"]]),
        ([["a"], ["b", "a"]], [["a"], ["b"]]),
        ([["future", "a"], ["future", "a", "later"]], [["future", "a"], ["later"]]),
    ],
    ids=["later-entry", "same-timestamp", "future-dated"],
)
async def test_continuous_mode_outputs_each_entry_once(monkeypatch, polls, expected):
    """Every entry is output once, whatever order and dates later polls bring."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    dates = {
        "old": now,
        "new": now + timedelta(minutes=1),
        "a": now,
        "b": now,
        "future": now + timedelta(days=365),
        "later": now + timedelta(hours=1),
    }
    block = ReadRSSFeedBlock()
    parse_feed = AsyncMock(
        side_effect=[
            {"entries": [_feed_entry(title, dates[title]) for title in poll]}
            for poll in polls
        ]
    )
    sleep = AsyncMock(side_effect=[None] * (len(polls) - 1) + [asyncio.CancelledError])
    monkeypatch.setattr(block, "parse_feed", parse_feed)
    monkeypatch.setattr("backend.blocks.rss.sleep", sleep)

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        polling_rate=60,
        run_continuously=True,
    )
    polled = []
    with pytest.raises(asyncio.CancelledError):
        async for name, value in block.run(input_data):
            if name == "entries":
                polled.append([e.title for e in value])

    assert polled == expected


def _feed_entry(title: str, published: datetime) -> dict:
    return {
        "title": title,
        "link": f"https://example.com/{title}",
        "published_parsed": published.timetuple(),
    }
