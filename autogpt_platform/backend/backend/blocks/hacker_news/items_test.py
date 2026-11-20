"""Unit tests for the Hacker News Get Item block.

The block's own test_input/test_mock case mocks the API calls away; these
cover the requests, the fallback to HN's official API, turning comments off,
and errors. Comment order is covered in _thread_test.py.
"""

from typing import Any

import pytest

from backend.blocks.hacker_news import items
from backend.blocks.hacker_news._api import HackerNewsError
from backend.blocks.hacker_news.items import HackerNewsGetItemBlock
from backend.util.exceptions import BlockExecutionError, BlockInputError

THREAD = {
    "id": 8863,
    "type": "story",
    "author": "dhouston",
    "title": "My YC app: Dropbox - Throw away your USB drive",
    "points": 104,
    "story_id": 8863,
    "created_at_i": 1175714200,
    "children": [
        {
            "id": 8865,
            "author": "dhouston",
            "text": "oh, and a mac port is coming :)",
            "parent_id": 8863,
            "created_at_i": 1175714575,
            "children": [],
        }
    ],
}
LIVE = {"id": 8863, "type": "story", "by": "dhouston", "kids": [8865], "score": 104}


@pytest.fixture
def block() -> HackerNewsGetItemBlock:
    return HackerNewsGetItemBlock()


class _FakeApi:
    """Stands in for the HN API calls _fetch makes, and records them."""

    def __init__(self, thread: Any, live: Any, replies: dict[int, Any] | None = None):
        self.thread, self.live, self.replies = thread, live, replies or {}
        self.calls: list[tuple[str, Any]] = []

    async def get_thread(self, item_id: int):
        self.calls.append(("get_thread", item_id))
        return self.thread

    async def get_item(self, item_id: int):
        self.calls.append(("get_item", item_id))
        return self.live

    async def get_replies(self, kids: list[int], limit: int):
        self.calls.append(("get_replies", (kids, limit)))
        return self.replies

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "_FakeApi":
        for name in ("get_thread", "get_item", "get_replies"):
            monkeypatch.setattr(items, name, getattr(self, name))
        return self


@pytest.mark.asyncio
async def test_fetch_reads_both_apis_and_no_replies_when_hn_search_has_the_item(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = _FakeApi(THREAD, LIVE).install(monkeypatch)
    assert await HackerNewsGetItemBlock._fetch(8863, 200) == (THREAD, LIVE, {})
    assert sorted(fake.calls) == [("get_item", 8863), ("get_thread", 8863)]


@pytest.mark.asyncio
async def test_fetch_falls_back_to_the_official_api_for_replies(
    monkeypatch: pytest.MonkeyPatch,
):
    replies = {8865: {"id": 8865, "by": "dhouston", "text": "hi", "parent": 8863}}
    fake = _FakeApi(None, LIVE, replies).install(monkeypatch)
    assert await HackerNewsGetItemBlock._fetch(8863, 50) == (None, LIVE, replies)
    assert ("get_replies", ([8865], 50)) in fake.calls


@pytest.mark.parametrize(
    "live, max_replies", [(LIVE, 0), (None, 200)], ids=["comments off", "no item"]
)
@pytest.mark.asyncio
async def test_fetch_skips_replies_when_none_are_wanted_or_there_is_no_item(
    monkeypatch: pytest.MonkeyPatch, live: Any, max_replies: int
):
    fake = _FakeApi(None, live).install(monkeypatch)
    assert await HackerNewsGetItemBlock._fetch(8863, max_replies) == (None, live, {})
    assert all(name != "get_replies" for name, _ in fake.calls)


@pytest.mark.asyncio
async def test_item_with_comments(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetItemBlock
):
    limits = mock_fetch(monkeypatch, block, (THREAD, LIVE, {}))
    outputs = await collect(block, item="8863", max_comments=5)
    assert [name for name, _ in outputs] == [
        "item",
        "comments",
        "comment",
        "comment_count",
    ]
    assert outputs[0][1].title == "My YC app: Dropbox - Throw away your USB drive"
    assert outputs[2][1].text == "oh, and a mac port is coming :)"
    assert outputs[3][1] == 1
    assert limits == [(8863, 5)]


@pytest.mark.asyncio
async def test_item_points_and_count_are_live_but_comments_are_hn_search_s(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetItemBlock
):
    # Measured on a front-page story: live 205 points and 112 comments while
    # HN Search had 198 and 106.
    live = {**LIVE, "score": 205, "descendants": 112}
    mock_fetch(monkeypatch, block, (THREAD, live, {}))
    outputs = dict(await collect(block, item="8863"))
    assert (outputs["item"].points, outputs["item"].num_comments) == (205, 112)
    assert outputs["comment_count"] == 112
    assert [c.id for c in outputs["comments"]] == [8865]


@pytest.mark.asyncio
async def test_item_without_comments_still_counts_them(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetItemBlock
):
    limits = mock_fetch(monkeypatch, block, (THREAD, LIVE, {}))
    outputs = await collect(block, item="8863", include_comments=False)
    assert [name for name, _ in outputs] == ["item", "comment_count"]
    assert outputs[1][1] == 1
    assert limits == [(8863, 0)]


@pytest.mark.asyncio
async def test_item_hn_search_has_not_indexed_comes_from_the_official_api(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetItemBlock
):
    live = {
        "id": 49965665,
        "type": "story",
        "by": "newposter",
        "title": "Posted a minute ago",
        "url": "https://example.com/new",
        "score": 2,
        "descendants": 1,
        "time": 1791212400,
        "kids": [49965700],
    }
    reply = {
        "id": 49965700,
        "type": "comment",
        "by": "fast",
        "text": "First!",
        "parent": 49965665,
        "time": 1791212460,
    }
    mock_fetch(monkeypatch, block, (None, live, {49965700: reply}))
    outputs = dict(await collect(block, item="news.ycombinator.com/item?id=49965665"))
    assert outputs["item"].title == "Posted a minute ago"
    assert outputs["item"].num_comments == 1
    assert [(c.id, c.author, c.depth) for c in outputs["comments"]] == [
        (49965700, "fast", 1)
    ]
    assert outputs["comment_count"] == 1


@pytest.mark.asyncio
async def test_unknown_item(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetItemBlock
):
    mock_fetch(monkeypatch, block, (None, None, {}))
    with pytest.raises(BlockExecutionError, match="Hacker News has no item 999999999"):
        await collect(block, item="999999999")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "abc",
        "https://example.com/item?id=1",
        "https://news.ycombinator.com/user?id=pg",
    ],
)
@pytest.mark.asyncio
async def test_input_that_is_not_an_item(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetItemBlock, text: str
):
    limits = mock_fetch(monkeypatch, block, (THREAD, LIVE, {}))
    with pytest.raises(BlockInputError, match="isn't a Hacker News item"):
        await collect(block, item=text)
    assert limits == []


@pytest.mark.asyncio
async def test_item_turns_api_errors_into_block_errors(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetItemBlock
):
    async def fake_fetch(item_id: int, max_replies: int):
        raise HackerNewsError("Hacker News search (hn.algolia.com) is limiting")

    monkeypatch.setattr(block, "_fetch", fake_fetch)
    with pytest.raises(BlockExecutionError, match="is limiting"):
        await collect(block, item="8863")


def mock_fetch(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetItemBlock, result: tuple
) -> list[tuple[int, int]]:
    """Make the block's _fetch return `result`; returns the (id, limit) it gets."""
    calls: list[tuple[int, int]] = []

    async def fake_fetch(item_id: int, max_replies: int):
        calls.append((item_id, max_replies))
        return result

    monkeypatch.setattr(block, "_fetch", fake_fetch)
    return calls


async def collect(
    block: HackerNewsGetItemBlock, **inputs: Any
) -> list[tuple[str, Any]]:
    return [output async for output in block.run(block.Input.model_validate(inputs))]
