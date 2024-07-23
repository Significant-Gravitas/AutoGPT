import time
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import pydantic

from autogpt_server.data.block import Block, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


class RSSEntry(pydantic.BaseModel):
    title: str
    link: str
    description: str
    pub_date: datetime
    author: str
    categories: list[str]


class RSSReaderBlock(Block):
    class Input(BlockSchema):
        rss_url: str = SchemaField(
            description="The URL of the RSS feed to read",
            placeholder="https://example.com/rss",
        )
        start_datetime: datetime = SchemaField(
            description="The date and time to start looking for posts from on the first loop. Defaults to 1 day ago.",
            placeholder="2023-06-23T12:00:00Z",
            default=datetime.now(timezone.utc) - timedelta(days=1),
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

    def __init__(self):
        super().__init__(
            id="c6731acb-4105-4zp1-bc9b-03d0036h370g",
            input_schema=RSSReaderBlock.Input,
            output_schema=RSSReaderBlock.Output,
            test_input={
                "rss_url": "https://example.com/rss",
                "start_datetime": "2023-06-01T12:00:00Z",
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
        return feedparser.parse(url)  # type: ignore

    def run(self, input_data: Input) -> BlockOutput:
        keep_going = True
        while keep_going:
            keep_going = input_data.run_continuously

            feed = self.parse_feed(input_data.rss_url)

            for entry in feed["entries"]:
                pub_date = datetime(*entry["published_parsed"][:6], tzinfo=timezone.utc)

                # if pub_date > input_data.start_datetime:
                yield (
                    "entry",
                    RSSEntry(
                        title=entry["title"],
                        link=entry["link"],
                        description=entry.get("summary", ""),
                        pub_date=pub_date,
                        author=entry.get("author", ""),
                        categories=[tag["term"] for tag in entry.get("tags", [])],
                    ),
                )

            time.sleep(input_data.polling_rate)
