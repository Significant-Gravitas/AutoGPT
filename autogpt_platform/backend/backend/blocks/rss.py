import asyncio
import logging
from asyncio import sleep as async_sleep
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any

import feedparser
import pydantic

from backend.blocks._base import (
    DEFAULT_BLOCK_EXECUTION_TIMEOUT_SECONDS,
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.request import Requests

DEFAULT_CONTINUOUS_RUNTIME_SECONDS = 60 * 60
MAX_CONTINUOUS_RUNTIME_SECONDS = 4 * 60 * 60


class RSSEntry(pydantic.BaseModel):
    title: str
    link: str
    description: str
    pub_date: datetime
    author: str
    categories: list[str]


class ReadRSSFeedBlock(Block):
    execution_timeout_seconds: int | None = (
        MAX_CONTINUOUS_RUNTIME_SECONDS + DEFAULT_BLOCK_EXECUTION_TIMEOUT_SECONDS
    )

    class Input(BlockSchemaInput):
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
            ge=1,
        )
        max_duration_seconds: int = SchemaField(
            description="Maximum time to poll continuously before this block finishes.",
            default=DEFAULT_CONTINUOUS_RUNTIME_SECONDS,
            ge=1,
            le=MAX_CONTINUOUS_RUNTIME_SECONDS,
        )
        run_continuously: bool = SchemaField(
            description="Whether to run the block continuously or just once.",
            default=True,
        )

    class Output(BlockSchemaOutput):
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
            effect=BlockEffect.READ,
        )

    @staticmethod
    async def parse_feed(url: str) -> dict[str, Any]:
        # Security fix: Add protection against memory exhaustion attacks
        MAX_FEED_SIZE = 10 * 1024 * 1024  # 10MB limit for RSS feeds

        # Download feed content with size limit
        try:
            response = await Requests(raise_for_status=True).get(url)

            # Check content length if available
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_FEED_SIZE:
                raise ValueError(
                    f"Feed too large: {content_length} bytes exceeds {MAX_FEED_SIZE} limit"
                )

            # Get content with size limit
            content = response.content
            if len(content) > MAX_FEED_SIZE:
                raise ValueError(f"Feed too large: exceeds {MAX_FEED_SIZE} byte limit")

            # Parse with feedparser using the validated content
            # feedparser has built-in protection against XML attacks
            return feedparser.parse(content)  # type: ignore
        except Exception as e:
            # Log error and return empty feed
            logging.warning(f"Failed to parse RSS feed from {url}: {e}")
            return {"entries": []}

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        start_time = datetime.now(timezone.utc) - timedelta(
            minutes=input_data.time_period
        )
        seen_links_at_start_time: set[str] = set()
        deadline = monotonic() + input_data.max_duration_seconds

        while True:
            if input_data.run_continuously and monotonic() >= deadline:
                break

            if input_data.run_continuously:
                feed = await self.parse_feed(input_data.rss_url)
            else:
                feed = await asyncio.wait_for(
                    self.parse_feed(input_data.rss_url),
                    timeout=DEFAULT_BLOCK_EXECUTION_TIMEOUT_SECONDS,
                )
            all_entries = []
            poll_start_time = start_time
            latest_pub_date = start_time
            latest_links_at_time = set(seen_links_at_start_time)
            seen_links_this_poll: set[str] = set()

            for entry in feed["entries"]:
                pub_date = datetime(*entry["published_parsed"][:6], tzinfo=timezone.utc)

                if pub_date < poll_start_time:
                    continue

                rss_entry = RSSEntry(
                    title=entry["title"],
                    link=entry["link"],
                    description=entry.get("summary", ""),
                    pub_date=pub_date,
                    author=entry.get("author", ""),
                    categories=[tag["term"] for tag in entry.get("tags", [])],
                )
                if rss_entry.link in seen_links_this_poll:
                    continue
                if (
                    pub_date == poll_start_time
                    and rss_entry.link in seen_links_at_start_time
                ):
                    continue

                seen_links_this_poll.add(rss_entry.link)
                all_entries.append(rss_entry)
                yield "entry", rss_entry

                if pub_date > latest_pub_date:
                    latest_pub_date = pub_date
                    latest_links_at_time = {rss_entry.link}
                elif pub_date == latest_pub_date:
                    latest_links_at_time.add(rss_entry.link)

            start_time = latest_pub_date
            seen_links_at_start_time = latest_links_at_time

            yield "entries", all_entries
            if not input_data.run_continuously:
                break

            remaining_seconds = deadline - monotonic()
            if remaining_seconds <= 0:
                break
            await async_sleep(min(input_data.polling_rate, remaining_seconds))
