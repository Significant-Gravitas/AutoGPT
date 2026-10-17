from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import feedparser

from backend.blocks.rss import ReadRSSFeedBlock

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


async def test_entries_missing_fields_do_not_abort_block():
    outputs = await run_block(RSS_WITH_MISSING_FIELDS)

    assert [e.title for e in outputs["entry"]] == ["Dated", "", "No link"]
    assert [e.link for e in outputs["entry"]] == [
        "https://example.com/dated",
        "https://example.com/untitled",
        "",
    ]
    assert outputs["entries"] == outputs["entry"]


async def test_atom_entry_with_only_updated_uses_updated_date():
    outputs = await run_block(ATOM_UPDATED_ONLY)

    assert len(outputs["entries"]) == 1
    assert outputs["entries"][0].title == "Updated only"
    assert outputs["entries"][0].pub_date == datetime(
        2023, 6, 23, 12, 30, 0, tzinfo=timezone.utc
    )
