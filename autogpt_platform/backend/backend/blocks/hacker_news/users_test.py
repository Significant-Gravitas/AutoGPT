"""Unit tests for the Hacker News Get User block.

The block's own test_input/test_mock case mocks the API call away; these
cover profiles with missing fields, unknown users and bad input.
"""

from typing import Any

import pytest

from backend.blocks.hacker_news import users
from backend.blocks.hacker_news._api import HackerNewsError
from backend.blocks.hacker_news.users import HackerNewsGetUserBlock
from backend.util.exceptions import BlockExecutionError, BlockInputError


@pytest.fixture
def block() -> HackerNewsGetUserBlock:
    return HackerNewsGetUserBlock()


@pytest.mark.asyncio
async def test_fetch_user_asks_for_the_username(monkeypatch: pytest.MonkeyPatch):
    asked: list[str] = []

    async def fake_get_user(username: str) -> dict[str, Any] | None:
        asked.append(username)
        return {"id": username}

    monkeypatch.setattr(users, "get_user", fake_get_user)
    assert await HackerNewsGetUserBlock._fetch_user("dang") == {"id": "dang"}
    assert asked == ["dang"]


@pytest.mark.asyncio
async def test_profile(monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetUserBlock):
    asked = mock_fetch(
        monkeypatch,
        block,
        {
            "id": "pg",
            "created": 1160418092,
            "karma": 157316,
            "about": "Bug fixer.",
            "submitted": [1, 2, 3, 4],
        },
    )
    assert await collect(block, username=" pg ") == [
        ("username", "pg"),
        ("karma", 157316),
        ("created_at", "2006-10-09T18:21:32Z"),
        ("about", "Bug fixer."),
        ("submission_count", 4),
        ("profile_url", "https://news.ycombinator.com/user?id=pg"),
    ]
    assert asked == ["pg"]


@pytest.mark.asyncio
async def test_profile_with_nothing_but_an_id(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetUserBlock
):
    mock_fetch(monkeypatch, block, {"id": "quiet"})
    assert dict(await collect(block, username="quiet")) == {
        "username": "quiet",
        "karma": 0,
        "about": "",
        "submission_count": 0,
        "profile_url": "https://news.ycombinator.com/user?id=quiet",
    }


@pytest.mark.asyncio
async def test_unknown_user(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetUserBlock
):
    mock_fetch(monkeypatch, block, None)
    with pytest.raises(BlockExecutionError, match="no user named 'PG'.*case-sensitive"):
        await collect(block, username="PG")


@pytest.mark.parametrize(
    "text", ["", "pg dang", "../item/1", "https://example.com/user?id=pg"]
)
@pytest.mark.asyncio
async def test_input_that_is_not_a_username(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetUserBlock, text: str
):
    asked = mock_fetch(monkeypatch, block, {"id": "x"})
    with pytest.raises(BlockInputError, match="isn't a Hacker News username"):
        await collect(block, username=text)
    assert asked == []


@pytest.mark.asyncio
async def test_api_errors_become_block_errors(
    monkeypatch: pytest.MonkeyPatch, block: HackerNewsGetUserBlock
):
    async def fake_fetch(username: str):
        raise HackerNewsError("The Hacker News API had a temporary problem")

    monkeypatch.setattr(block, "_fetch_user", fake_fetch)
    with pytest.raises(BlockExecutionError, match="temporary problem"):
        await collect(block, username="pg")


def mock_fetch(
    monkeypatch: pytest.MonkeyPatch,
    block: HackerNewsGetUserBlock,
    user: dict[str, Any] | None,
) -> list[str]:
    """Make the block's _fetch_user return `user`; returns the usernames it gets."""
    asked: list[str] = []

    async def fake_fetch(username: str):
        asked.append(username)
        return user

    monkeypatch.setattr(block, "_fetch_user", fake_fetch)
    return asked


async def collect(
    block: HackerNewsGetUserBlock, **inputs: Any
) -> list[tuple[str, Any]]:
    return [output async for output in block.run(block.Input.model_validate(inputs))]
