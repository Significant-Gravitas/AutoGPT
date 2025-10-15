import asyncio
import logging
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import pydantic

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class RSSEntry(pydantic.BaseModel):
    title: str
    link: str
    description: str
    pub_date: datetime
    author: str
    categories: list[str]


class ReadRSSFeedBlock(Block):
    class Input(BlockSchema):
        rss_url: str = SchemaField(
            description="The URL of the RSS feed to read",
            placeholder="https://example.com/rss",
        )
        time_period: int = SchemaField(
            description="The time period to check in minutes relative to the run block runtime, e.g. 60 would check for new entries in the last hour.",
            placeholder="1440",
            default=1440,
        )
        polling_rate: int = SchemaField(
            description="The number of seconds to wait between polling attempts.",
            placeholder="300",
        )
        run_continuously: bool = SchemaField(
            description="Whether to run the block continuously or just once.",
            default=True,
        )

    class Output(BlockSchema):
        entry: RSSEntry = SchemaField(description="The RSS item")
        entries: list[RSSEntry] = SchemaField(description="List of all RSS entries")

    def __init__(self):
        super().__init__(
            id="5ebe6768-8e5d-41e3-9134-1c7bd89a8d52",
            input_schema=ReadRSSFeedBlock.Input,
            output_schema=ReadRSSFeedBlock.Output,
            description="Reads RSS feed entries from a given URL.",
            categories={BlockCategory.INPUT},
            test_input={
                "rss_url": "https://example.com/rss",
                "time_period": 10_000_000,
                "polling_rate": 1,
                "run_continuously": False,
            },
            test_output=[
                (
                    "entry",
                    RSSEntry(
                        title="Example RSS Item",
                        link="https://example.com/article",
                        description="This is an example RSS item description.",
                        pub_date=datetime(2023, 6, 23, 12, 30, 0, tzinfo=timezone.utc),
                        author="John Doe",
                        categories=["Technology", "News"],
                    ),
                ),
                (
                    "entries",
                    [
                        RSSEntry(
                            title="Example RSS Item",
                            link="https://example.com/article",
                            description="This is an example RSS item description.",
                            pub_date=datetime(
                                2023, 6, 23, 12, 30, 0, tzinfo=timezone.utc
                            ),
                            author="John Doe",
                            categories=["Technology", "News"],
                        ),
                    ],
                ),
            ],
            test_mock={
                "parse_feed": lambda *args, **kwargs: {
                    "entries": [
                        {
                            "title": "Example RSS Item",
                            "link": "https://example.com/article",
                            "summary": "This is an example RSS item description.",
                            "published_parsed": (2023, 6, 23, 12, 30, 0, 4, 174, 0),
                            "author": "John Doe",
                            "tags": [{"term": "Technology"}, {"term": "News"}],
                        }
                    ]
                }
            },
        )

    @staticmethod
    def parse_feed(url: str) -> dict[str, Any]:
        # Security fix: Add protection against memory exhaustion attacks
        MAX_FEED_SIZE = 10 * 1024 * 1024  # 10MB limit for RSS feeds

        # Validate URL
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}")

        # Download with size limit
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                # Check content length if available
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_FEED_SIZE:
                    raise ValueError(
                        f"Feed too large: {content_length} bytes exceeds {MAX_FEED_SIZE} limit"
                    )

                # Read with size limit
                content = response.read(MAX_FEED_SIZE + 1)
                if len(content) > MAX_FEED_SIZE:
                    raise ValueError(
                        f"Feed too large: exceeds {MAX_FEED_SIZE} byte limit"
                    )

                # Parse with feedparser using the validated content
                # feedparser has built-in protection against XML attacks
                return feedparser.parse(content)  # type: ignore
        except Exception as e:
            # Log error and return empty feed
            logging.warning(f"Failed to parse RSS feed from {url}: {e}")
            return {"entries": []}

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        keep_going = True
        start_time = datetime.now(timezone.utc) - timedelta(
            minutes=input_data.time_period
        )
        while keep_going:
            keep_going = input_data.run_continuously

            feed = self.parse_feed(input_data.rss_url)
            all_entries = []

            for entry in feed["entries"]:
                pub_date = datetime(*entry["published_parsed"][:6], tzinfo=timezone.utc)

                if pub_date > start_time:
                    rss_entry = RSSEntry(
                        title=entry["title"],
                        link=entry["link"],
                        description=entry.get("summary", ""),
                        pub_date=pub_date,
                        author=entry.get("author", ""),
                        categories=[tag["term"] for tag in entry.get("tags", [])],
                    )
                    all_entries.append(rss_entry)
                    yield "entry", rss_entry

            yield "entries", all_entries
            await asyncio.sleep(input_data.polling_rate)
