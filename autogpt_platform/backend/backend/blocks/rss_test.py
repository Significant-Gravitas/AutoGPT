import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, call, patch

import feedparser
import pytest

from backend.blocks.rss import ReadRSSFeedBlock, RSSEntry

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
    undated = {
        "title": "No date",
        "link": "https://example.com/undated",
    }
    parse_feed = AsyncMock(
        return_value={"entries": [entry, undated] if has_entries else []}
    )
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
