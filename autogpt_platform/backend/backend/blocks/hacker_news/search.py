import re
from datetime import datetime, timedelta, timezone
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
from backend.util.exceptions import BlockExecutionError, BlockInputError

from ._api import HackerNewsError, parse_username, search
from ._models import HackerNewsItem, item_from_hit


class SearchFor(str, Enum):
    STORIES = "stories"
    COMMENTS = "comments"
    STORIES_AND_COMMENTS = "stories_and_comments"
    SHOW_HN = "show_hn"
    ASK_HN = "ask_hn"
    FRONT_PAGE = "front_page"


class MatchIn(str, Enum):
    ANYWHERE = "anywhere"
    TITLE = "title"
    URL = "url"


class SortBy(str, Enum):
    RELEVANCE = "relevance"
    NEWEST = "newest"


# HN Search tags: a comma means AND, brackets mean OR.
_TAGS = {
    SearchFor.STORIES: "story",
    SearchFor.COMMENTS: "comment",
    SearchFor.STORIES_AND_COMMENTS: "(story,comment)",
    SearchFor.SHOW_HN: "show_hn",
    SearchFor.ASK_HN: "ask_hn",
    SearchFor.FRONT_PAGE: "front_page",
}
_AGE = re.compile(r"(\d{1,5})\s*(h|hours?|d|days?|w|weeks?)", re.IGNORECASE)
_HOURS_PER_UNIT = {"h": 1, "d": 24, "w": 24 * 7}

_TEST_STORY_HIT: dict[str, Any] = {
    "_tags": ["story", "author_builder42", "story_49900001", "show_hn"],
    "objectID": "49900001",
    "title": "Show HN: Hacker News blocks for AutoGPT",
    "url": "https://agpt.co/blog/hacker-news-blocks",
    "author": "builder42",
    "points": 42,
    "num_comments": 7,
    "created_at_i": 1790780400,
    "story_id": 49900001,
}
_TEST_COMMENT_HIT: dict[str, Any] = {
    "_tags": ["comment", "author_hn_reader", "story_49900001"],
    "objectID": "49900107",
    "comment_text": (
        "I&#x27;ve tried <i>AutoGPT</i> for this.<p>Docs: "
        '<a href="https:&#x2F;&#x2F;agpt.co&#x2F;docs&#x2F;integrations" '
        'rel="nofollow">https:&#x2F;&#x2F;agpt.co&#x2F;docs&#x2F;...</a>'
    ),
    "author": "hn_reader",
    "points": None,
    "story_id": 49900001,
    "story_title": "Show HN: Hacker News blocks for AutoGPT",
    "story_url": "https://agpt.co/blog/hacker-news-blocks",
    "parent_id": 49900001,
    "created_at_i": 1790785800,
}


class HackerNewsSearchBlock(Block):
    """Search Hacker News stories and comments through HN Search (Algolia)."""

    class Input(BlockSchemaInput):
        query: str = SchemaField(
            description=(
                "Words to search for. Put a phrase in double quotes to match it as "
                "a phrase, and put - before a word to leave out items that have it. "
                "Leave empty to match everything, e.g. for the top stories of the "
                "past week."
            ),
            default="",
            placeholder='e.g. "AutoGPT" or agpt.co',
        )
        search_for: SearchFor = SchemaField(
            description=(
                "What to search: stories, comments, stories_and_comments, show_hn or "
                "ask_hn posts, or front_page (only the stories on the front page now)"
            ),
            default=SearchFor.STORIES,
        )
        match_in: MatchIn = SchemaField(
            description=(
                "Where the query must match: anywhere (title, link, text and author), "
                "title, or url (the link a story points to, e.g. to find links to a "
                "domain). title and url only match stories."
            ),
            default=MatchIn.ANYWHERE,
        )
        exact_match: bool = SchemaField(
            description=(
                "Match the query words as written, without typo tolerance. On by "
                "default, because typo tolerance makes 'autogpt' also match "
                "'automotive' and 'automation'. Plurals still match. Turn it off to "
                "also catch misspellings."
            ),
            default=True,
        )
        sort_by: SortBy = SchemaField(
            description=(
                "relevance: best matches first, then most points, then most comments. "
                "newest: newest first."
            ),
            default=SortBy.RELEVANCE,
        )
        created_after: str = SchemaField(
            description=(
                "Only items posted at or after this time: an ISO 8601 date or time "
                "(2026-09-01, 2026-09-01T12:00:00Z; UTC unless it has an offset), or "
                "an age such as 24h, 7d or 2w"
            ),
            default="",
            placeholder="e.g. 7d or 2026-09-01",
        )
        created_before: str = SchemaField(
            description=(
                "Only items posted before this time, written the same way as "
                "created_after"
            ),
            default="",
            placeholder="e.g. 2026-10-01",
        )
        author: str = SchemaField(
            description="Only items posted by this Hacker News username (case-sensitive)",
            default="",
            placeholder="e.g. pg",
            advanced=True,
        )
        min_points: int = SchemaField(
            description=(
                "Only stories with at least this many points. Comments have no "
                "points, so any value above 0 leaves comments out."
            ),
            default=0,
            ge=0,
            advanced=True,
        )
        min_comments: int = SchemaField(
            description=(
                "Only stories with at least this many comments. Any value above 0 "
                "leaves comments out."
            ),
            default=0,
            ge=0,
            advanced=True,
        )
        max_results: int = SchemaField(
            description="Most results to return (up to 1,000)",
            default=20,
            ge=1,
            le=1000,
        )

    class Output(BlockSchemaOutput):
        results: list[HackerNewsItem] = SchemaField(
            description="Matching stories and comments, in the chosen order"
        )
        result: HackerNewsItem = SchemaField(
            description="Each matching story or comment"
        )
        total_matches: int = SchemaField(
            description=(
                "How many items match in all. Can be more than the results returned: "
                "HN Search returns at most 1,000."
            )
        )

    def __init__(self):
        story = item_from_hit(_TEST_STORY_HIT)
        comment = item_from_hit(_TEST_COMMENT_HIT)
        super().__init__(
            id="f86d4a13-e470-4a7c-9eba-2bb6a76c0552",
            description=(
                "Search Hacker News stories and comments by keyword, author, date, "
                "points or linked domain, sorted by relevance or newest first. Use it "
                "to find mentions of a product, links to a website, or the most "
                "upvoted stories of the week. Uses HN Search by Algolia, which needs "
                "no account. exact_match is on by default, so words that are only "
                "spelled alike are left out; turn it off to also catch misspellings."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.SOCIAL},
            input_schema=HackerNewsSearchBlock.Input,
            output_schema=HackerNewsSearchBlock.Output,
            test_input={
                "query": "AutoGPT",
                "search_for": SearchFor.STORIES_AND_COMMENTS,
                "sort_by": SortBy.NEWEST,
            },
            test_output=[
                ("results", [comment, story]),
                ("result", comment),
                ("result", story),
                ("total_matches", 2),
            ],
            test_mock={
                "_search": lambda *args, **kwargs: {
                    "hits": [_TEST_COMMENT_HIT, _TEST_STORY_HIT],
                    "nbHits": 2,
                }
            },
            effect=BlockEffect.READ,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        params = self.search_params(input_data, datetime.now(timezone.utc))
        endpoint = "search_by_date" if input_data.sort_by == SortBy.NEWEST else "search"
        try:
            response = await self._search(endpoint, params)
        except HackerNewsError as e:
            raise BlockExecutionError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e

        results = [item_from_hit(hit) for hit in response.get("hits") or []]
        yield "results", results
        for result in results:
            yield "result", result
        yield "total_matches", response.get("nbHits", len(results))

    @staticmethod
    async def _search(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        return await search(endpoint, params)

    def search_params(self, input_data: Input, now: datetime) -> dict[str, str]:
        """HN Search query parameters for the block's inputs."""
        if (
            input_data.search_for == SearchFor.COMMENTS
            and input_data.match_in != MatchIn.ANYWHERE
        ):
            raise self._input_error(
                "Comments have no title or link of their own, so they can only be "
                "matched anywhere. Set match_in to anywhere, or search stories."
            )
        tags = _TAGS[input_data.search_for]
        if input_data.author.strip():
            tags += f",author_{self._username(input_data.author)}"
        params = {
            "query": input_data.query.strip(),
            "tags": tags,
            "hitsPerPage": str(input_data.max_results),
        }
        if filters := self._numeric_filters(input_data, now):
            params["numericFilters"] = filters
        if input_data.match_in != MatchIn.ANYWHERE:
            params["restrictSearchableAttributes"] = input_data.match_in.value
        if input_data.exact_match:
            params["typoTolerance"] = "false"
        return params

    def _numeric_filters(self, input_data: Input, now: datetime) -> str:
        after = self._time(input_data.created_after, "created_after", now)
        before = self._time(input_data.created_before, "created_before", now)
        if after is not None and before is not None and after >= before:
            raise self._input_error(
                "created_after must be earlier than created_before."
            )
        filters: list[str] = []
        if after is not None:
            filters.append(f"created_at_i>={after}")
        if before is not None:
            filters.append(f"created_at_i<{before}")
        if input_data.min_points:
            filters.append(f"points>={input_data.min_points}")
        if input_data.min_comments:
            filters.append(f"num_comments>={input_data.min_comments}")
        return ",".join(filters)

    def _time(self, text: str, field: str, now: datetime) -> int | None:
        try:
            return parse_time(text, now)
        except ValueError as e:
            raise self._input_error(
                f"{field} '{text.strip()}' isn't a date or an age. Use an ISO 8601 "
                "date or time such as 2026-09-01 or 2026-09-01T12:00:00Z, or an age "
                "such as 24h, 7d or 2w."
            ) from e

    def _username(self, text: str) -> str:
        username = parse_username(text)
        if not username:
            raise self._input_error(
                f"'{text.strip()}' isn't a Hacker News username. Usernames have only "
                "letters, digits, - and _."
            )
        return username

    def _input_error(self, message: str) -> BlockInputError:
        return BlockInputError(message=message, block_name=self.name, block_id=self.id)


def parse_time(text: str, now: datetime) -> int | None:
    """Unix seconds for an ISO 8601 date or time, or for an age such as 24h, 7d
    or 2w before `now`. None for empty text; ValueError for anything else.

    A date or time without a UTC offset is taken as UTC.
    """
    text = text.strip()
    if not text:
        return None
    if age := _AGE.fullmatch(text):
        hours = int(age[1]) * _HOURS_PER_UNIT[age[2][0].lower()]
        return int((now - timedelta(hours=hours)).timestamp())
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return int(moment.timestamp())
