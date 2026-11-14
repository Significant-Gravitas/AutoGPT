"""Unit tests for the Hacker News API helpers: requests, error messages,
concurrent item fetches, and reading ids and usernames from text."""

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.blocks.hacker_news import _api
from backend.blocks.hacker_news._api import (
    ALGOLIA_URL,
    FIREBASE_URL,
    MAX_CONCURRENT_REQUESTS,
    HackerNewsError,
    parse_item_id,
    parse_username,
)
from backend.util.request import Response


def _response(status: int, body: Any = None, reason: str = "Reason") -> Response:
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    raw = MagicMock(status=status, headers={}, reason=reason)
    return Response(response=raw, url="https://example.invalid", body=content)


class _FakeRequests:
    """Stands in for the Requests class: records its options and each GET."""

    def __init__(self, *responses: Response):
        self.responses = list(responses)
        self.options: list[dict[str, Any]] = []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, **kwargs: Any) -> "_FakeRequests":
        self.options.append(kwargs)
        return self

    async def get(self, url: str, **kwargs: Any) -> Response:
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


@pytest.fixture
def fake_requests(monkeypatch: pytest.MonkeyPatch):
    def install(*responses: Response) -> _FakeRequests:
        fake = _FakeRequests(*responses)
        monkeypatch.setattr(_api, "Requests", fake)
        return fake

    return install


@pytest.mark.asyncio
async def test_search_calls_hn_search_with_the_params(fake_requests):
    fake = fake_requests(_response(200, {"hits": [], "nbHits": 0}))
    params = {"query": "agpt.co", "tags": "story"}
    assert await _api.search("search_by_date", params) == {"hits": [], "nbHits": 0}
    assert fake.calls == [(f"{ALGOLIA_URL}/search_by_date", {"params": params})]
    # Only the two Hacker News hosts skip the SSRF IP checks, and 429/5xx
    # responses are retried twice, then reported rather than raised as-is.
    assert fake.options == [
        {
            "trusted_origins": [
                "https://hn.algolia.com",
                "https://hacker-news.firebaseio.com",
            ],
            "raise_for_status": False,
            "retry_max_attempts": 3,
        }
    ]


@pytest.mark.asyncio
async def test_item_user_and_list_urls(fake_requests):
    fake = fake_requests(
        _response(200, {"id": 8863}),
        _response(200, {"id": 8863, "children": []}),
        _response(200, {"id": "pg"}),
        _response(200, [3, 2, 1]),
    )
    assert await _api.get_item(8863) == {"id": 8863}
    assert await _api.get_thread(8863) == {"id": 8863, "children": []}
    assert await _api.get_user("pg") == {"id": "pg"}
    assert await _api.get_story_ids("jobstories") == [3, 2, 1]
    assert [url for url, _ in fake.calls] == [
        f"{FIREBASE_URL}/item/8863.json",
        f"{ALGOLIA_URL}/items/8863",
        f"{FIREBASE_URL}/user/pg.json",
        f"{FIREBASE_URL}/jobstories.json",
    ]


@pytest.mark.asyncio
async def test_missing_items_and_users_come_back_as_none(fake_requests):
    # HN Search answers 404; the official API answers 200 with `null`.
    fake_requests(
        _response(404, {"error": "Not Found", "status": 404}),
        _response(200, b"null"),
        _response(200, b"null"),
        _response(200, b"null"),
    )
    assert await _api.get_thread(1) is None
    assert await _api.get_item(1) is None
    assert await _api.get_user("nobody") is None
    assert await _api.get_story_ids("topstories") == []


@pytest.mark.parametrize(
    "url, status, body, expected",
    [
        (
            f"{ALGOLIA_URL}/search",
            429,
            None,
            "Hacker News search (hn.algolia.com) is limiting how many requests",
        ),
        (
            f"{FIREBASE_URL}/item/1.json",
            503,
            b"<html>Unavailable</html>",
            "The Hacker News API had a temporary problem (HTTP 503)",
        ),
        (
            f"{ALGOLIA_URL}/search",
            400,
            {"message": "invalid setting for restrictSearchableAttributes"},
            "rejected the request (HTTP 400): invalid setting for "
            "restrictSearchableAttributes",
        ),
        (f"{FIREBASE_URL}/item/1.json", 401, b"", "(HTTP 401): Permission denied"),
    ],
)
@pytest.mark.asyncio
async def test_errors_become_messages_a_user_can_act_on(
    fake_requests, url: str, status: int, body: Any, expected: str
):
    fake_requests(_response(status, body, reason="Permission denied"))
    with pytest.raises(HackerNewsError) as error:
        await _api.get_json(url)
    assert expected in str(error.value)


@pytest.mark.asyncio
async def test_get_items_keeps_order_and_limits_concurrency(
    monkeypatch: pytest.MonkeyPatch,
):
    running = peak = 0

    async def fake_get_item(item_id: int) -> dict[str, Any] | None:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.001 * (item_id % 4))  # finish out of order
        running -= 1
        return None if item_id == 7 else {"id": item_id}

    monkeypatch.setattr(_api, "get_item", fake_get_item)
    items = await _api.get_items(list(range(30)))
    assert [item["id"] if item else None for item in items] == [
        None if item_id == 7 else item_id for item_id in range(30)
    ]
    assert peak == MAX_CONCURRENT_REQUESTS


@pytest.mark.parametrize(
    "limit, batches, found",
    [
        (10, [[1, 2], [11, 12, 21], [111]], {1, 2, 11, 21, 111}),
        # 12 answers null, so the last slot goes to 21, still a level up from 111.
        (4, [[1, 2], [11, 12], [21]], {1, 2, 11, 21}),
        (3, [[1, 2], [11]], {1, 2, 11}),
    ],
)
@pytest.mark.asyncio
async def test_get_replies_fetches_breadth_first_up_to_the_limit(
    monkeypatch: pytest.MonkeyPatch,
    limit: int,
    batches: list[list[int]],
    found: set[int],
):
    kids = {1: [11, 12], 2: [21], 11: [111], 21: [], 111: []}
    requested: list[list[int]] = []

    async def fake_get_items(item_ids: list[int]) -> list[dict[str, Any] | None]:
        requested.append(item_ids)
        return [
            {"id": item_id, "kids": kids[item_id]} if item_id in kids else None
            for item_id in item_ids
        ]

    monkeypatch.setattr(_api, "get_items", fake_get_items)
    replies = await _api.get_replies([1, 2], limit=limit)
    assert requested == batches
    assert set(replies) == found


@pytest.mark.asyncio
async def test_get_replies_fetches_each_id_once_even_in_a_loop(
    monkeypatch: pytest.MonkeyPatch,
):
    # Malformed data: 1 is listed twice, and 2 and 3 list each other.
    kids = {1: [2], 2: [3], 3: [2, 1]}
    requested: list[int] = []

    async def fake_get_items(item_ids: list[int]) -> list[dict[str, Any] | None]:
        requested.extend(item_ids)
        return [{"id": item_id, "kids": kids[item_id]} for item_id in item_ids]

    monkeypatch.setattr(_api, "get_items", fake_get_items)
    replies = await _api.get_replies([1, 1], limit=100)
    assert sorted(requested) == [1, 2, 3]
    assert set(replies) == {1, 2, 3}


@pytest.mark.asyncio
async def test_get_replies_with_no_comments_makes_no_requests(
    monkeypatch: pytest.MonkeyPatch,
):
    fake = MagicMock()
    monkeypatch.setattr(_api, "get_items", fake)
    assert await _api.get_replies([], limit=200) == {}
    fake.assert_not_called()


@pytest.mark.parametrize(
    "text, expected",
    [
        ("8863", 8863),
        (" 008863 ", 8863),
        ("https://news.ycombinator.com/item?id=8863", 8863),
        ("http://news.ycombinator.com/item?id=8863&p=2", 8863),
        ("news.ycombinator.com/item?id=8863", 8863),
        ("https://news.ycombinator.com/item?id=8863#8865", 8863),
        ("https://news.ycombinator.com/user?id=8863", None),
        ("https://example.com/item?id=8863", None),
        ("https://news.ycombinator.com/item?id=abc", None),
        ("https://news.ycombinator.com/item", None),
        ("https://[news.ycombinator.com/item?id=1", None),
        ("-5", None),
        ("８８６３", None),  # full-width digits
        ("", None),
    ],
)
def test_parse_item_id(text: str, expected: int | None):
    assert parse_item_id(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("pg", "pg"),
        (" dang ", "dang"),
        ("some_user-1", "some_user-1"),
        ("https://news.ycombinator.com/user?id=pg", "pg"),
        ("news.ycombinator.com/user?id=tptacek", "tptacek"),
        ("https://news.ycombinator.com/item?id=pg", None),
        ("https://example.com/user?id=pg", None),
        ("pg,author_dang", None),
        ("../item/1", None),
        ("a" * 33, None),
        ("", None),
    ],
)
def test_parse_username(text: str, expected: str | None):
    assert parse_username(text) == expected
