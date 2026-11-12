"""Unit tests for the Hacker News Search block: query parameters, tag and
numeric filter strings, dates, input errors and outputs.

The block's own test_input/test_mock case mocks the API call away; these
cover what that mock skips.
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from backend.blocks._base import BlockOutput
from backend.blocks.hacker_news._api import HackerNewsError
from backend.blocks.hacker_news.search import HackerNewsSearchBlock, parse_time
from backend.util.exceptions import BlockExecutionError, BlockInputError

NOW = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)
NOW_TS = int(NOW.timestamp())
HOUR = 3600
DAY = 24 * HOUR


def params(**inputs: Any) -> dict[str, str]:
    block = HackerNewsSearchBlock()
    return block.search_params(block.Input.model_validate(inputs), NOW)


def test_defaults_search_stories_without_typo_tolerance():
    assert params() == {
        "query": "",
        "tags": "story",
        "hitsPerPage": "20",
        "typoTolerance": "false",
    }


def test_typo_tolerance_can_be_turned_back_on():
    assert "typoTolerance" not in params(query="autogtp", exact_match=False)


@pytest.mark.parametrize(
    "search_for, author, tags",
    [
        ("stories", "", "story"),
        ("comments", "", "comment"),
        ("stories_and_comments", "", "(story,comment)"),
        ("show_hn", "", "show_hn"),
        ("ask_hn", "", "ask_hn"),
        ("front_page", "", "front_page"),
        ("stories", "pg", "story,author_pg"),
        ("stories_and_comments", " dang ", "(story,comment),author_dang"),
        (
            "comments",
            "https://news.ycombinator.com/user?id=tptacek",
            "comment,author_tptacek",
        ),
    ],
)
def test_tags(search_for: str, author: str, tags: str):
    assert params(search_for=search_for, author=author)["tags"] == tags


def test_domain_search_matches_story_links_exactly():
    assert params(query=" agpt.co ", match_in="url", max_results=1000) == {
        "query": "agpt.co",
        "tags": "story",
        "hitsPerPage": "1000",
        "restrictSearchableAttributes": "url",
        "typoTolerance": "false",
    }


def test_title_search_on_stories_and_comments_is_allowed():
    found = params(search_for="stories_and_comments", match_in="title")
    assert found["restrictSearchableAttributes"] == "title"


@pytest.mark.parametrize("match_in", ["title", "url"])
def test_comments_cannot_be_matched_by_title_or_link(match_in: str):
    with pytest.raises(BlockInputError, match="Set match_in to anywhere"):
        params(search_for="comments", match_in=match_in)


def test_numeric_filters_combine_with_commas():
    found = params(
        created_after="2025-09-01",
        created_before="2025-10-01T00:00:00Z",
        min_points=100,
        min_comments=10,
    )
    assert found["numericFilters"] == (
        "created_at_i>=1756684800,created_at_i<1759276800,"
        "points>=100,num_comments>=10"
    )


@pytest.mark.parametrize(
    "created_after, created_at_or_after",
    [
        ("24h", NOW_TS - DAY),
        ("7d", NOW_TS - 7 * DAY),
        ("2w", NOW_TS - 14 * DAY),
        ("3 days", NOW_TS - 3 * DAY),
        ("1 Hour", NOW_TS - HOUR),
    ],
)
def test_relative_ages_count_back_from_now(
    created_after: str, created_at_or_after: int
):
    found = params(created_after=created_after)
    assert found["numericFilters"] == f"created_at_i>={created_at_or_after}"


def test_zero_minimums_add_no_filter():
    assert "numericFilters" not in params(min_points=0, min_comments=0)


@pytest.mark.parametrize(
    "field, value",
    [
        ("created_after", "yesterday"),
        ("created_after", "7x"),
        ("created_before", "2026-13-01"),
        ("created_before", "1/2/2026"),
    ],
)
def test_bad_dates_name_the_input(field: str, value: str):
    with pytest.raises(BlockInputError, match=f"{field} '{value}' isn't a date"):
        params(**{field: value})


def test_dates_must_be_in_order():
    with pytest.raises(BlockInputError, match="earlier than created_before"):
        params(created_after="2026-09-02", created_before="2026-09-01")


@pytest.mark.parametrize("author", ["pg,dang", "(pg)", "pg dang"])
def test_bad_authors_are_refused_before_they_reach_the_tags(author: str):
    with pytest.raises(BlockInputError, match="isn't a Hacker News username"):
        params(author=author)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2026-09-01", 1788220800),
        ("2026-09-01T12:00:00Z", 1788264000),
        ("2026-09-01T12:00:00", 1788264000),
        ("2026-09-01T14:00:00+02:00", 1788264000),
        ("  48h ", NOW_TS - 2 * DAY),
        ("0d", NOW_TS),
        ("", None),
        ("   ", None),
    ],
)
def test_parse_time(text: str, expected: int | None):
    assert parse_time(text, NOW) == expected


@pytest.mark.parametrize("text", ["last week", "7", "d7", "-7d", "123456d"])
def test_parse_time_rejects_anything_else(text: str):
    with pytest.raises(ValueError):
        parse_time(text, NOW)


STORY_HIT = {
    "_tags": ["story", "author_pg", "story_1"],
    "objectID": "1",
    "title": "Y Combinator",
    "url": "http://ycombinator.com",
    "author": "pg",
    "points": 57,
    "num_comments": 18,
    "created_at_i": 1160418111,
    "story_id": 1,
}


@pytest.mark.parametrize(
    "sort_by, endpoint", [("relevance", "search"), ("newest", "search_by_date")]
)
@pytest.mark.asyncio
async def test_run_searches_and_outputs_each_result(
    monkeypatch: pytest.MonkeyPatch, sort_by: str, endpoint: str
):
    block = HackerNewsSearchBlock()
    calls: list[tuple[str, dict[str, str]]] = []

    async def fake_search(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        calls.append((endpoint, params))
        return {"hits": [STORY_HIT], "nbHits": 4321}

    monkeypatch.setattr(block, "_search", fake_search)
    outputs = await collect(block, query="ycombinator", sort_by=sort_by)
    assert [name for name, _ in outputs] == ["results", "result", "total_matches"]
    assert outputs[0][1][0].title == "Y Combinator"
    assert outputs[1][1] == outputs[0][1][0]
    assert outputs[2][1] == 4321
    assert calls == [
        (
            endpoint,
            {
                "query": "ycombinator",
                "tags": "story",
                "hitsPerPage": "20",
                "typoTolerance": "false",
            },
        )
    ]


@pytest.mark.asyncio
async def test_run_with_no_matches(monkeypatch: pytest.MonkeyPatch):
    block = HackerNewsSearchBlock()

    async def fake_search(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        return {"hits": [], "nbHits": 0}

    monkeypatch.setattr(block, "_search", fake_search)
    assert await collect(block, query="zzzzqqqq") == [
        ("results", []),
        ("total_matches", 0),
    ]


@pytest.mark.asyncio
async def test_run_turns_api_errors_into_block_errors(monkeypatch: pytest.MonkeyPatch):
    block = HackerNewsSearchBlock()

    async def fake_search(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        raise HackerNewsError("Hacker News search (hn.algolia.com) had a problem")

    monkeypatch.setattr(block, "_search", fake_search)
    with pytest.raises(BlockExecutionError, match="had a problem"):
        await collect(block, query="x")


async def collect(block: HackerNewsSearchBlock, **inputs: Any) -> list[tuple[str, Any]]:
    outputs: BlockOutput = block.run(block.Input.model_validate(inputs))
    return [output async for output in outputs]
