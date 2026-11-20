import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, call, patch

import feedparser
import pytest

from backend.blocks._base import DEFAULT_BLOCK_EXECUTION_TIMEOUT_SECONDS
from backend.blocks.rss import (
    MAX_CONTINUOUS_RUNTIME_SECONDS,
    ReadRSSFeedBlock,
    RSSEntry,
)

RSS_WITH_MISSING_FIELDS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Feed</title>
<item><title>Dated</title><link>https://example.com/dated</link>
<pubDate>Fri, 23 Jun 2023 12:30:00 GMT</pubDate></item>
<item><title>No date</title><link>https://example.com/undated</link></item>
<item><link>https://example.com/untitled</link>
<pubDate>Sat, 24 Jun 2023 12:30:00 GMT</pubDate></item>
<item><title>No link</title>
<pubDate>Sun, 25 Jun 2023 12:30:00 GMT</pubDate></item>
</channel></rss>"""

ATOM_UPDATED_ONLY = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Feed</title>
<entry><title>Updated only</title>
<link href="https://example.com/updated"/>
<updated>2023-06-23T12:30:00Z</updated></entry>
</feed>"""

ATOM_CATEGORY_WITHOUT_TERM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Feed</title>
<entry><title>Categories</title>
<link href="https://example.com/categories"/>
<updated>2023-06-23T12:30:00Z</updated>
<category label="label only"/><category term="news"/></entry>
</feed>"""


def make_feed_entry(now: datetime, title: str, link: str, minutes_ago: int) -> dict:
    published = now - timedelta(minutes=minutes_ago)
    return {
        "title": title,
        "link": link,
        "published_parsed": published.utctimetuple(),
    }


def make_expected_entry(
    now: datetime, title: str, link: str, minutes_ago: int
) -> RSSEntry:
    return RSSEntry(
        title=title,
        link=link,
        description="",
        pub_date=now - timedelta(minutes=minutes_ago),
        author="",
        categories=[],
    )


def patch_continuous_polling(monkeypatch, block, feed_pages: list) -> None:
    parse_feed = AsyncMock(side_effect=feed_pages)
    sleep = AsyncMock(side_effect=[None, asyncio.CancelledError])
    monkeypatch.setattr(block, "parse_feed", parse_feed)
    monkeypatch.setattr("backend.blocks.rss.async_sleep", sleep)
    monkeypatch.setattr("backend.blocks.rss.monotonic", lambda: 0)


def continuous_input(**overrides) -> ReadRSSFeedBlock.Input:
    params = {
        "rss_url": "https://example.com/rss",
        "time_period": 60,
        "polling_rate": 42,
        "max_duration_seconds": 3600,
        "run_continuously": True,
    }
    return ReadRSSFeedBlock.Input(**(params | overrides))


async def collect_until_cancelled(block, input_data) -> list:
    outputs = []
    with pytest.raises(asyncio.CancelledError):
        async for output in block.run(input_data):
            outputs.append(output)
    return outputs


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
    undated = {
        "title": "No date",
        "link": "https://example.com/undated",
    }
    parse_feed = AsyncMock(
        return_value={"entries": [entry, undated] if has_entries else []}
    )
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

    first_entry = make_feed_entry(now, "First", "https://example.com/first", 10)
    second_entry = make_feed_entry(now, "Second", "https://example.com/second", 5)
    third_entry = make_feed_entry(now, "Third", "https://example.com/third", 4)
    patch_continuous_polling(
        monkeypatch,
        block,
        [
            {"entries": [first_entry, second_entry]},
            {"entries": [second_entry, first_entry, third_entry]},
        ],
    )

    outputs = await collect_until_cancelled(block, continuous_input())

    expected_first = make_expected_entry(now, "First", first_entry["link"], 10)
    expected_second = make_expected_entry(now, "Second", second_entry["link"], 5)
    expected_third = make_expected_entry(now, "Third", third_entry["link"], 4)
    assert outputs == [
        ("entry", expected_first),
        ("entry", expected_second),
        ("entries", [expected_first, expected_second]),
        ("entry", expected_third),
        ("entries", [expected_third]),
    ]


async def test_polling_rate_zero_is_clamped_to_one_second(monkeypatch):
    """Saved run-once workflows hold polling_rate=0 and must stay valid."""
    block = ReadRSSFeedBlock()
    parse_feed = AsyncMock(return_value={"entries": []})
    sleep = AsyncMock(side_effect=[None, asyncio.CancelledError])
    monkeypatch.setattr(block, "parse_feed", parse_feed)
    monkeypatch.setattr("backend.blocks.rss.async_sleep", sleep)
    monkeypatch.setattr("backend.blocks.rss.monotonic", lambda: 0)

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        polling_rate=0,
        run_continuously=True,
    )
    with pytest.raises(asyncio.CancelledError):
        async for _ in block.run(input_data):
            pass

    assert sleep.await_args_list == [call(1), call(1)]


async def test_continuous_mode_emits_late_entries_with_older_pub_dates(monkeypatch):
    """The window cutoff stays fixed, so a late-arriving entry with an older
    pub_date is still emitted; only links already emitted this run are skipped."""
    block = ReadRSSFeedBlock()
    now = datetime.now(timezone.utc).replace(microsecond=0)

    recent_entry = make_feed_entry(now, "Recent", "https://example.com/recent", 10)
    late_entry = make_feed_entry(now, "Late", "https://example.com/late", 45)
    patch_continuous_polling(
        monkeypatch,
        block,
        [
            {"entries": [recent_entry]},
            {"entries": [recent_entry, late_entry]},
        ],
    )

    outputs = await collect_until_cancelled(block, continuous_input())

    expected_recent = make_expected_entry(now, "Recent", recent_entry["link"], 10)
    expected_late = make_expected_entry(now, "Late", late_entry["link"], 45)
    assert outputs == [
        ("entry", expected_recent),
        ("entries", [expected_recent]),
        ("entry", expected_late),
        ("entries", [expected_late]),
    ]


async def test_continuous_mode_fetch_timeout_finishes_normally(monkeypatch):
    """A fetch that outlasts the remaining duration ends the run, not fails it."""
    block = ReadRSSFeedBlock()

    async def hanging_parse_feed(url: str) -> dict:
        await asyncio.sleep(60)
        return {"entries": []}

    monkeypatch.setattr(block, "parse_feed", hanging_parse_feed)
    monkeypatch.setattr("backend.blocks.rss.monotonic", lambda: 0)

    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        polling_rate=42,
        max_duration_seconds=1,
        run_continuously=True,
    )
    outputs = [output async for output in block.run(input_data)]

    assert outputs == []


async def test_entries_missing_fields_do_not_abort_block():
    outputs = await run_block(RSS_WITH_MISSING_FIELDS)

    assert [e.title for e in outputs["entry"]] == ["Dated", "", "No link"]
    assert [e.link for e in outputs["entry"]] == [
        "https://example.com/dated",
        "https://example.com/untitled",
        "",
    ]
    assert outputs["entries"] == outputs["entry"]


async def test_skipped_undated_entries_are_logged(caplog):
    with caplog.at_level("WARNING"):
        await run_block(RSS_WITH_MISSING_FIELDS)

    assert "Skipped 1 RSS feed entries without a date" in caplog.text
    assert "example.com" not in caplog.text


async def test_atom_entry_with_only_updated_uses_updated_date():
    outputs = await run_block(ATOM_UPDATED_ONLY)

    assert len(outputs["entries"]) == 1
    assert outputs["entries"][0].title == "Updated only"
    assert outputs["entries"][0].pub_date == datetime(
        2023, 6, 23, 12, 30, 0, tzinfo=timezone.utc
    )


async def test_category_without_term_is_ignored():
    outputs = await run_block(ATOM_CATEGORY_WITHOUT_TERM)

    assert outputs["entries"][0].categories == ["news"]


async def run_block(feed_xml: str) -> dict:
    block = ReadRSSFeedBlock()
    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        time_period=10_000_000,
        polling_rate=0,
        run_continuously=False,
    )
    feed = feedparser.parse(feed_xml)
    outputs: dict = {"entry": []}
    with patch.object(ReadRSSFeedBlock, "parse_feed", new=AsyncMock(return_value=feed)):
        async for name, value in block.run(input_data):
            if name == "entry":
                outputs["entry"].append(value)
            else:
                outputs[name] = value
    return outputs


def test_rss_feed_block_has_finite_wall_clock_timeout():
    assert ReadRSSFeedBlock().execution_timeout_seconds == (
        MAX_CONTINUOUS_RUNTIME_SECONDS + DEFAULT_BLOCK_EXECUTION_TIMEOUT_SECONDS
    )


async def test_rss_feed_block_skips_final_sleep_for_one_shot(monkeypatch):
    sleep_calls = []

    async def parse_feed(url):
        return {"entries": []}

    async def sleep(delay):
        sleep_calls.append(delay)

    monkeypatch.setattr(ReadRSSFeedBlock, "parse_feed", staticmethod(parse_feed))
    monkeypatch.setattr("backend.blocks.rss.async_sleep", sleep)

    block = ReadRSSFeedBlock()
    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        time_period=60,
        polling_rate=300,
        run_continuously=False,
    )
    outputs = [output async for output in block.run(input_data)]

    assert outputs == [("entries", [])]
    assert sleep_calls == []


async def test_rss_feed_block_times_out_one_shot_fetch(monkeypatch):
    async def parse_feed(url):
        await asyncio.sleep(1)

    monkeypatch.setattr(ReadRSSFeedBlock, "parse_feed", staticmethod(parse_feed))
    monkeypatch.setattr("backend.blocks.rss.DEFAULT_BLOCK_EXECUTION_TIMEOUT_SECONDS", 0)

    block = ReadRSSFeedBlock()
    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        time_period=60,
        polling_rate=300,
        run_continuously=False,
    )

    with pytest.raises(TimeoutError):
        async for _ in block.run(input_data):
            pass


async def test_rss_feed_block_keeps_polling_for_continuous_reads(monkeypatch):
    sleep_calls = []
    outputs = []

    class StopPolling(Exception):
        pass

    async def parse_feed(url):
        return {"entries": []}

    async def sleep(delay):
        sleep_calls.append(delay)
        raise StopPolling

    monkeypatch.setattr(ReadRSSFeedBlock, "parse_feed", staticmethod(parse_feed))
    monkeypatch.setattr("backend.blocks.rss.async_sleep", sleep)

    block = ReadRSSFeedBlock()
    input_data = ReadRSSFeedBlock.Input(
        rss_url="https://example.com/rss",
        time_period=60,
        polling_rate=300,
        run_continuously=True,
    )

    with pytest.raises(StopPolling):
        async for output in block.run(input_data):
            outputs.append(output)

    assert outputs == [("entries", [])]
    assert sleep_calls == [300]
