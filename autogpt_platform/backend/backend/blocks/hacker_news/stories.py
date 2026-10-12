from enum import Enum
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
from backend.util.exceptions import BlockExecutionError

from ._api import HackerNewsError, get_items, get_story_ids
from ._models import HackerNewsStory, story_from_firebase


class StoryList(str, Enum):
    TOP = "top"
    NEW = "new"
    BEST = "best"
    ASK = "ask"
    SHOW = "show"
    JOBS = "jobs"


_LIST_NAMES = {
    StoryList.TOP: "topstories",
    StoryList.NEW: "newstories",
    StoryList.BEST: "beststories",
    StoryList.ASK: "askstories",
    StoryList.SHOW: "showstories",
    StoryList.JOBS: "jobstories",
}

_TEST_STORY: dict[str, Any] = {
    "id": 49963366,
    "type": "story",
    "by": "builder42",
    "time": 1791201600,
    "title": "Show HN: Hacker News blocks for AutoGPT",
    "url": "https://agpt.co/blog/hacker-news-blocks",
    "score": 224,
    "descendants": 37,
    "kids": [49963489, 49963513],
}
_TEST_DEAD_STORY: dict[str, Any] = {"id": 49963400, "type": "story", "dead": True}
_TEST_JOB: dict[str, Any] = {
    "id": 49945904,
    "type": "job",
    "by": "acme_hiring",
    "time": 1791205800,
    "title": "Acme (YC W24) Is Hiring Engineers",
    "url": "https://www.ycombinator.com/companies/acme/jobs",
    "score": 1,
}


class HackerNewsGetStoriesBlock(Block):
    """The stories on one of Hacker News's lists, such as the front page."""

    class Input(BlockSchemaInput):
        story_list: StoryList = SchemaField(
            description=(
                "Which list: top (the front page ranking), new, best (most upvoted "
                "lately), ask (Ask HN), show (Show HN) or jobs"
            ),
            default=StoryList.TOP,
        )
        max_results: int = SchemaField(
            description=(
                "Most stories to return, from the top of the list. Top, new and best "
                "hold up to 500 stories; ask, show and jobs up to 200."
            ),
            default=30,
            ge=1,
            le=500,
        )

    class Output(BlockSchemaOutput):
        stories: list[HackerNewsStory] = SchemaField(
            description="The stories, in the list's order"
        )
        story: HackerNewsStory = SchemaField(description="Each story")

    def __init__(self):
        story = story_from_firebase(_TEST_STORY, rank=1)
        job = story_from_firebase(_TEST_JOB, rank=3)
        super().__init__(
            id="dda11fa8-5d5f-4254-a4b6-1d23afffea3b",
            description=(
                "Get the top (front page), new, best, Ask HN, Show HN or job stories "
                "from Hacker News, in the order Hacker News ranks them. Each story "
                "comes with its rank, title, link, points, comment count and poster. "
                "Uses Hacker News's official API, which needs no account."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.SOCIAL},
            input_schema=HackerNewsGetStoriesBlock.Input,
            output_schema=HackerNewsGetStoriesBlock.Output,
            test_input={"story_list": StoryList.TOP, "max_results": 3},
            test_output=[
                ("stories", [story, job]),
                ("story", story),
                ("story", job),
            ],
            test_mock={
                "_fetch_stories": lambda *args, **kwargs: [
                    _TEST_STORY,
                    _TEST_DEAD_STORY,
                    _TEST_JOB,
                ]
            },
            effect=BlockEffect.READ,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            items = await self._fetch_stories(
                input_data.story_list, input_data.max_results
            )
        except HackerNewsError as e:
            raise BlockExecutionError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e

        # Rank is the place on HN's list, so it skips the items left out here.
        stories = [
            story_from_firebase(item, rank)
            for rank, item in enumerate(items, start=1)
            if item and not item.get("deleted") and not item.get("dead")
        ]
        yield "stories", stories
        for story in stories:
            yield "story", story

    @staticmethod
    async def _fetch_stories(
        story_list: StoryList, limit: int
    ) -> list[dict[str, Any] | None]:
        """The first `limit` items on the list. The official API has one URL
        per item, so this is one request per story, 10 at a time."""
        story_ids = await get_story_ids(_LIST_NAMES[story_list])
        return await get_items(story_ids[:limit])
