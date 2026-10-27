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

from ._api import HackerNewsError, get_user, parse_username, user_url
from ._html import html_to_text
from ._models import iso_time

_TEST_USER: dict[str, Any] = {
    "id": "builder42",
    "created": 1160418092,
    "karma": 1337,
    "about": (
        "Builds agents.<p>Blog: "
        '<a href="https:&#x2F;&#x2F;agpt.co">https:&#x2F;&#x2F;agpt.co</a>'
    ),
    "submitted": [49900215, 49900001, 49871234],
}


class HackerNewsGetUserBlock(Block):
    """A Hacker News user's public profile."""

    class Input(BlockSchemaInput):
        username: str = SchemaField(
            description=(
                "The username, such as pg (case-sensitive), or a profile link such "
                "as https://news.ycombinator.com/user?id=pg"
            ),
            placeholder="e.g. pg",
        )

    class Output(BlockSchemaOutput):
        username: str = SchemaField(
            description="The username, spelled the way Hacker News has it"
        )
        karma: int = SchemaField(description="The user's karma")
        created_at: str = SchemaField(
            description="When the account was created, in ISO 8601 (UTC)"
        )
        about: str = SchemaField(
            description="The user's about text as plain text; empty if they wrote none"
        )
        submission_count: int = SchemaField(
            description="How many stories, comments and polls the user has posted"
        )
        profile_url: str = SchemaField(description="Link to the profile on Hacker News")

    def __init__(self):
        super().__init__(
            id="7c78fe61-29e0-45fe-9613-888ec40ee349",
            description=(
                "Get a Hacker News user's profile by username: karma, when the "
                "account was created, the about text and how many items they have "
                "posted. Uses Hacker News's official API, which needs no account."
            ),
            categories={BlockCategory.SEARCH, BlockCategory.SOCIAL},
            input_schema=HackerNewsGetUserBlock.Input,
            output_schema=HackerNewsGetUserBlock.Output,
            test_input={"username": "https://news.ycombinator.com/user?id=builder42"},
            test_output=[
                ("username", "builder42"),
                ("karma", 1337),
                ("created_at", "2006-10-09T18:21:32Z"),
                ("about", "Builds agents.\n\nBlog: https://agpt.co"),
                ("submission_count", 3),
                ("profile_url", "https://news.ycombinator.com/user?id=builder42"),
            ],
            test_mock={"_fetch_user": lambda *args, **kwargs: _TEST_USER},
            effect=BlockEffect.READ,
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        username = parse_username(input_data.username)
        if not username:
            raise BlockInputError(
                message=(
                    f"'{input_data.username.strip()}' isn't a Hacker News username. "
                    "Usernames have only letters, digits, - and _."
                ),
                block_name=self.name,
                block_id=self.id,
            )
        try:
            user = await self._fetch_user(username)
        except HackerNewsError as e:
            raise BlockExecutionError(
                message=str(e), block_name=self.name, block_id=self.id
            ) from e
        if user is None:
            raise BlockExecutionError(
                message=(
                    f"Hacker News has no user named '{username}'. Usernames are "
                    "case-sensitive, so check the capitals."
                ),
                block_name=self.name,
                block_id=self.id,
            )

        yield "username", user.get("id") or username
        yield "karma", user.get("karma") or 0
        if created_at := iso_time(user.get("created")):
            yield "created_at", created_at
        yield "about", html_to_text(user.get("about"))
        yield "submission_count", len(user.get("submitted") or [])
        yield "profile_url", user_url(user.get("id") or username)

    @staticmethod
    async def _fetch_user(username: str) -> dict[str, Any] | None:
        return await get_user(username)
