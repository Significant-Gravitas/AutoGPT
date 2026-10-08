import asyncio
from typing import Any

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField
from backend.util.exceptions import BlockExecutionError, BlockInputError

from ._api import HackerNewsError, get_item, get_replies, get_thread, parse_item_id
from ._models import HackerNewsComment, HackerNewsItem
from ._thread import read_live, read_thread

_TEST_THREAD: dict[str, Any] = {
    "id": 49900001,
    "type": "story",
    "author": "builder42",
    "title": "Show HN: Hacker News blocks for AutoGPT",
    "url": "https://agpt.co/blog/hacker-news-blocks",
    "text": None,
    "points": 42,
    "parent_id": None,
    "story_id": 49900001,
    "created_at_i": 1790780400,
    "children": [
        {
            "id": 49900107,
            "author": "hn_reader",
            "text": "Does it read <i>Ask HN</i> posts too?",
            "parent_id": 49900001,
            "created_at_i": 1790781600,
            "children": [
                {
                    "id": 49900215,
                    "author": "builder42",
                    "text": "Yes, as plain text.",
                    "parent_id": 49900107,
                    "created_at_i": 1790783100,
                    "children": [],
                }
            ],
        },
        {
            # Deleted: HN Search keeps it, without author or text, for its reply.
            "id": 49900150,
            "author": None,
            "text": None,
            "parent_id": 49900001,
            "created_at_i": 1790782000,
            "children": [
                {
                    "id": 49900301,
                    "author": "lurker",
                    "text": "What did it say?",
                    "parent_id": 49900150,
                    "created_at_i": 1790787900,
                    "children": [],
                }
            ],
        },
    ],
}
# The same story from HN's official API. It ranks the deleted comment first,
# and has 3 more points and 1 more comment than HN Search has indexed so far.
_TEST_LIVE_ITEM: dict[str, Any] = {
    "id": 49900001,
    "type": "story",
    "by": "builder42",
    "time": 1790780400,
    "score": 45,
    "descendants": 4,
    "kids": [49900150, 49900107],
}


class HackerNewsGetItemBlock(Block):
    """A Hacker News story, comment, poll or job, with its comments."""

    class Input(BlockSchemaInput):
        item: str = SchemaField(
            description=(
                "The item's id, such as 8863, or its link, such as "
                "https://news.ycombinator.com/item?id=8863"
            ),
            placeholder="e.g. 8863",
        )
        include_comments: bool = SchemaField(
            description="Also return the item's comments", default=True
        )
        max_comments: int = SchemaField(
            description=(
                "Most comments to return, in reading order. comment_count still "
                "counts them all."
            ),
            default=200,
            ge=1,
            le=1000,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        item: HackerNewsItem = SchemaField(
            description="The story, comment, poll or job"
        )
        comments: list[HackerNewsComment] = SchemaField(
            description=(
                "Its comments in reading order: each comment followed by its "
                "replies, with direct replies in Hacker News's ranking"
            )
        )
        comment: HackerNewsComment = SchemaField(description="Each comment")
        comment_count: int = SchemaField(
            description=(
                "How many comments the item has, including any that max_comments "
                "left out. Hacker News's own count when it is higher."
            )
        )

    def __init__(self):
        item = HackerNewsItem(
            id=49900001,
            type="story",
            title="Show HN: Hacker News blocks for AutoGPT",
            url="https://agpt.co/blog/hacker-news-blocks",
            author="builder42",
            points=45,
            num_comments=4,
            created_at="2026-09-30T15:00:00Z",
            hn_url="https://news.ycombinator.com/item?id=49900001",
            story_id=49900001,
        )
        # The deleted comment is ranked first: it is left out, its reply kept.
        comments = [
            HackerNewsComment(
                id=49900301,
                author="lurker",
                text="What did it say?",
                created_at="2026-09-30T17:05:00Z",
                parent_id=49900150,
                depth=2,
                hn_url="https://news.ycombinator.com/item?id=49900301",
            ),
            HackerNewsComment(
                id=49900107,
                author="hn_reader",
                text="Does it read *Ask HN* posts too?",
                created_at="2026-09-30T15:20:00Z",
                parent_id=49900001,
                depth=1,
                hn_url="https://news.ycombinator.com/item?id=49900107",
            ),
        ]
        super().__init__(
            id="3150fc1d-f7bc-46be-bdbe-2650ff7b2d62",
            description=(
                "Get a Hacker News story, comment, poll or job by its id or link, with "
                "its comments as a list in reading order. Each comment is followed by "
                "its replies, and direct replies come in Hacker News's ranking."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.SOCIAL},
            input_schema=HackerNewsGetItemBlock.Input,
            output_schema=HackerNewsGetItemBlock.Output,
            test_input={
                "item": "https://news.ycombinator.com/item?id=49900001",
                "max_comments": 2,
            },
            test_output=[
                ("item", item),
                ("comments", comments),
                ("comment", comments[0]),
                ("comment", comments[1]),
                ("comment_count", 4),
            ],
            test_mock={
                "_fetch": lambda *args, **kwargs: (_TEST_THREAD, _TEST_LIVE_ITEM, {})
            },
            effect=BlockEffect.READ,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        item_id = parse_item_id(input_data.item)
        if item_id is None:
            raise BlockInputError(
                message=(
                    f"'{input_data.item.strip()}' isn't a Hacker News item. Give its "
                    "id, such as 8863, or its link, such as "
                    "https://news.ycombinator.com/item?id=8863."
                ),
                block_name=self.name,
                block_id=self.id,
            )
        limit = input_data.max_comments if input_data.include_comments else 0
        try:
            thread, live, replies = await self._fetch(item_id, limit)
        except HackerNewsError as e:
            raise BlockExecutionError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e

        if thread is not None:
            item, comments, count = read_thread(thread, live, limit)
        elif live is not None:
            item, comments, count = read_live(live, replies, limit)
        else:
            raise BlockExecutionError(
                message=f"Hacker News has no item {item_id}. Check the id or link.",
                block_name=self.name,
                block_id=self.id,
            )
        yield "item", item
        if input_data.include_comments:
            yield "comments", comments
            for comment in comments:
                yield "comment", comment
        yield "comment_count", count

    @staticmethod
    async def _fetch(
        item_id: int, max_replies: int
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[int, dict[str, Any]]]:
        """The item's thread from HN Search and the item from HN's official API.

        HN Search doesn't have items from the last few minutes, or dead ones.
        For those, up to `max_replies` comments come from the official API.
        """
        thread, live = await asyncio.gather(get_thread(item_id), get_item(item_id))
        replies: dict[int, dict[str, Any]] = {}
        if thread is None and live is not None and max_replies:
            replies = await get_replies(live.get("kids") or [], max_replies)
        return thread, live, replies
