"""Unit tests for the Hacker News Get Stories block.

The block's own test_input/test_mock case mocks the API calls away; these
cover which list is read, how many items are fetched, the items left out, and
errors.
"""

from typing import Any

import pytest

from backend.blocks.hacker_news import stories
from backend.blocks.hacker_news._api import HackerNewsError
from backend.blocks.hacker_news.stories import HackerNewsGetStoriesBlock, StoryList
from backend.util.exceptions import BlockExecutionError


def story(item_id: int, **extra: Any) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "story",
        "by": "pg",
        "title": f"Story {item_id}",
        "score": 10,
        "descendants": 2,
        "time": 1160418111,
        **extra,
    }


@pytest.mark.parametrize(
    "story_list, list_name",
    [
        (StoryList.TOP, "topstories"),
        (StoryList.NEW, "newstories"),
        (StoryList.BEST, "beststories"),
        (StoryList.ASK, "askstories"),
        (StoryList.SHOW, "showstories"),
        (StoryList.JOBS, "jobstories"),
    ],
)
@pytest.mark.asyncio
async def test_fetch_stories_reads_the_list_then_its_first_items(
    monkeypatch: pytest.MonkeyPatch, story_list: StoryList, list_name: str
):
    requested: list[Any] = []

    async def fake_story_ids(name: str) -> list[int]:
        requested.append(name)
        return [5, 4, 3, 2, 1]

    async def fake_items(item_ids: list[int]) -> list[dict[str, Any] | None]:
        requested.append(item_ids)
        return [story(item_id) for item_id in item_ids]

    monkeypatch.setattr(stories, "get_story_ids", fake_story_ids)
    monkeypatch.setattr(stories, "get_items", fake_items)
    items = await HackerNewsGetStoriesBlock._fetch_stories(story_list, 3)
    assert requested == [list_name, [5, 4, 3]]
    assert [item["id"] for item in items if item] == [5, 4, 3]


@pytest.mark.asyncio
async def test_stories_skip_missing_deleted_and_dead_items_but_keep_hn_rank(
    monkeypatch: pytest.MonkeyPatch,
):
    block = HackerNewsGetStoriesBlock()

    async def fake_fetch(story_list: StoryList, limit: int):
        assert (story_list, limit) == (StoryList.NEW, 6)
        return [
            story(10),
            None,
            story(12, deleted=True),
            story(13, dead=True),
            {"id": 14, "type": "job", "by": "acme", "title": "Hiring", "score": 1},
            story(15),
        ]

    monkeypatch.setattr(block, "_fetch_stories", fake_fetch)
    outputs = await collect(block, story_list="new", max_results=6)
    found = outputs[0][1]
    assert outputs[0][0] == "stories"
    assert [(s.id, s.rank, s.type) for s in found] == [
        (10, 1, "story"),
        (14, 5, "job"),
        (15, 6, "story"),
    ]
    assert found[1].points is None
    assert outputs[1:] == [("story", s) for s in found]


@pytest.mark.asyncio
async def test_an_empty_list(monkeypatch: pytest.MonkeyPatch):
    block = HackerNewsGetStoriesBlock()

    async def fake_fetch(story_list: StoryList, limit: int):
        return []

    monkeypatch.setattr(block, "_fetch_stories", fake_fetch)
    assert await collect(block, story_list="jobs") == [("stories", [])]


@pytest.mark.asyncio
async def test_stories_turn_api_errors_into_block_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    block = HackerNewsGetStoriesBlock()

    async def fake_fetch(story_list: StoryList, limit: int):
        raise HackerNewsError("The Hacker News API had a temporary problem")

    monkeypatch.setattr(block, "_fetch_stories", fake_fetch)
    with pytest.raises(BlockExecutionError, match="temporary problem"):
        await collect(block)


async def collect(
    block: HackerNewsGetStoriesBlock, **inputs: Any
) -> list[tuple[str, Any]]:
    return [output async for output in block.run(block.Input.model_validate(inputs))]
