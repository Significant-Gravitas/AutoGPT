import time
from datetime import datetime, timezone

import feedparser

from autogpt_server.data.block import Block, BlockOutput, BlockSchema
from autogpt_server.data.model import SchemaField


class RSSReaderBlock(Block):
    class Input(BlockSchema):
        rss_url: str = SchemaField(
            description="The URL of the RSS feed to read",
            placeholder="https://example.com/rss",
        )
        start_datetime: datetime = SchemaField(
            description="The date and time to start looking for posts from on the first loop.",
            placeholder="2023-06-23T12:00:00Z",
            default="2024-07-19T16:14:32.707240Z",
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
        title: str = SchemaField(description="The title of the RSS item")
        link: str = SchemaField(description="The link to the full article")
        description: str = SchemaField(
            description="The description or summary of the RSS item"
        )
        pub_date: datetime = SchemaField(
            description="The publication date and time of the RSS item"
        )
        author: str = SchemaField(description="The author of the RSS item", default="")
        categories: list[str] = SchemaField(
            description="The categories or tags of the RSS item", default_factory=list
        )

    def __init__(self):
        super().__init__(
            id="c6731acb-4105-4zp1-bc9b-03d0036h370g",
            input_schema=RSSReaderBlock.Input,
            output_schema=RSSReaderBlock.Output,
            test_input={
                "rss_url": "https://example.com/rss",
                "start_datetime": "2023-06-23T12:00:00Z",
                "polling_rate": 300,
                "run_continuously": False,
            },
            test_output=[
                ("title", "Example RSS Item"),
                ("link", "https://example.com/article"),
                ("description", "This is an example RSS item description."),
                ("pub_date", datetime(2023, 6, 23, 12, 30, 0, tzinfo=timezone.utc)),
                ("author", "John Doe"),
                ("categories", ["Technology", "News"]),
            ],
            test_mock={
                "feedparser.parse": lambda url: {
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

    def run(self, input_data: Input) -> BlockOutput:
        while input_data.run_continuously:
            feed = feedparser.parse(input_data.rss_url)

            for entry in feed.entries:
                pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)

                if pub_date > input_data.start_datetime:
                    yield "title", entry.title
                    yield "link", entry.link
                    yield "description", entry.get("summary", "")
                    yield "pub_date", pub_date
                    yield "author", entry.get("author", "")
                    yield "categories", [tag.term for tag in entry.get("tags", [])]
                    return

            time.sleep(input_data.polling_rate)
