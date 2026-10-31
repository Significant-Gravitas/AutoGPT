import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, call

import pytest
from pydantic import ValidationError

from backend.blocks.rss import ReadRSSFeedBlock, RSSEntry


@pytest.mark.parametrize("has_entries", [False, True])
@pytest.mark.parametrize("polling_rate", [1, 60, 300])
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
    monkeypatch.setattr("backend.blocks.rss.async_sleep", sleep)

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
    monkeypatch.setattr("backend.blocks.rss.async_sleep", sleep)

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


async def test_continuous_mode_stops_at_max_duration(monkeypatch):
    block = ReadRSSFeedBlock()
    parse_feed = AsyncMock(return_value={"entries": []})
    sleep = AsyncMock()
    monotonic_values = iter([0, 0, 1])
    monkeypatch.setattr(block, "parse_feed", parse_feed)
    monkeypatch.setattr("backend.blocks.rss.async_sleep", sleep)
    monkeypatch.setattr("backend.blocks.rss.monotonic", lambda: next(monotonic_values))

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        polling_rate=42,
        max_duration_seconds=1,
        run_continuously=True,
    )
    outputs = [output async for output in block.run(input_data)]

    assert outputs == [("entries", [])]
    parse_feed.assert_awaited_once_with(input_data.rss_url)
    sleep.assert_not_awaited()


async def test_continuous_mode_emits_only_new_feed_entries(monkeypatch):
    block = ReadRSSFeedBlock()
    now = datetime.now(timezone.utc).replace(microsecond=0)

    def feed_entry(title: str, link: str, minutes_ago: int) -> dict:
        published = now - timedelta(minutes=minutes_ago)
        return {
            "title": title,
            "link": link,
            "published_parsed": published.utctimetuple(),
        }

    first_entry = feed_entry("First", "https://example.com/first", 10)
    second_entry = feed_entry("Second", "https://example.com/second", 5)
    third_entry = feed_entry("Third", "https://example.com/third", 4)
    parse_feed = AsyncMock(
        side_effect=[
            {"entries": [first_entry, second_entry]},
            {"entries": [second_entry, first_entry, third_entry]},
        ]
    )
    sleep = AsyncMock(side_effect=[None, asyncio.CancelledError])
    monkeypatch.setattr(block, "parse_feed", parse_feed)
    monkeypatch.setattr("backend.blocks.rss.async_sleep", sleep)
    monkeypatch.setattr("backend.blocks.rss.monotonic", lambda: 0)

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        time_period=60,
        polling_rate=42,
        max_duration_seconds=3600,
        run_continuously=True,
    )
    outputs = []
    with pytest.raises(asyncio.CancelledError):
        async for output in block.run(input_data):
            outputs.append(output)

    expected_first = RSSEntry(
        title="First",
        link=first_entry["link"],
        description="",
        pub_date=now - timedelta(minutes=10),
        author="",
        categories=[],
    )
    expected_second = RSSEntry(
        title="Second",
        link=second_entry["link"],
        description="",
        pub_date=now - timedelta(minutes=5),
        author="",
        categories=[],
    )
    expected_third = RSSEntry(
        title="Third",
        link=third_entry["link"],
        description="",
        pub_date=now - timedelta(minutes=4),
        author="",
        categories=[],
    )
    assert outputs == [
        ("entry", expected_first),
        ("entry", expected_second),
        ("entries", [expected_first, expected_second]),
        ("entry", expected_third),
        ("entries", [expected_third]),
    ]


def test_polling_rate_must_be_positive():
    with pytest.raises(ValidationError):
        ReadRSSFeedBlock.Input(
            rss_url="https://example.com/rss",
            polling_rate=0,
            run_continuously=True,
        )
