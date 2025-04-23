from typing import cast

import tweepy
from tweepy.client import Response

from backend.blocks.twitter._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    TwitterCredentials,
    TwitterCredentialsField,
    TwitterCredentialsInput,
)
from backend.blocks.twitter._builders import SpaceExpansionsBuilder
from backend.blocks.twitter._serializer import (
    IncludesSerializer,
    ResponseDataSerializer,
)
from backend.blocks.twitter._types import (
    SpaceExpansionInputs,
    SpaceExpansionsFilter,
    SpaceFieldsFilter,
    SpaceStatesFilter,
    TweetUserFieldsFilter,
)
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TwitterSearchSpacesBlock(Block):
    """
    Returns live or scheduled Spaces matching specified search terms [for a week only]
    """

    class Input(SpaceExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["spaces.read", "users.read", "tweet.read", "offline.access"]
        )

        query: str = SchemaField(
            description="Search term to find in Space titles",
            placeholder="Enter search query",
        )

        max_results: int | None = SchemaField(
            description="Maximum number of results to return (1-100)",
            placeholder="Enter max results",
            default=10,
            advanced=True,
        )

        state: SpaceStatesFilter = SchemaField(
            description="Type of Spaces to return (live, scheduled, or all)",
            placeholder="Enter state filter",
            default=SpaceStatesFilter.all,
        )

    class Output(BlockSchema):
        # Common outputs that user commonly uses
        ids: list[str] = SchemaField(description="List of space IDs")
        titles: list[str] = SchemaField(description="List of space titles")
        host_ids: list = SchemaField(description="List of host IDs")
        next_token: str = SchemaField(description="Next token for pagination")

        # Complete outputs for advanced use
        data: list[dict] = SchemaField(description="Complete space data")
        includes: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        meta: dict = SchemaField(description="Metadata including pagination info")

        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="aaefdd48-a62f-11ef-a73c-3f44df63e276",
            description="This block searches for Twitter Spaces based on specified terms.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterSearchSpacesBlock.Input,
            output_schema=TwitterSearchSpacesBlock.Output,
            test_input={
                "query": "tech",
                "max_results": 1,
                "state": "live",
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": None,
                "space_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["1234"]),
                ("titles", ["Tech Talk"]),
                ("host_ids", ["5678"]),
                ("data", [{"id": "1234", "title": "Tech Talk", "host_ids": ["5678"]}]),
            ],
            test_mock={
                "search_spaces": lambda *args, **kwargs: (
                    [{"id": "1234", "title": "Tech Talk", "host_ids": ["5678"]}],
                    {},
                    {},
                    ["1234"],
                    ["Tech Talk"],
                    ["5678"],
                    None,
                )
            },
        )

    @staticmethod
    def search_spaces(
        credentials: TwitterCredentials,
        query: str,
        max_results: int | None,
        state: SpaceStatesFilter,
        expansions: SpaceExpansionsFilter | None,
        space_fields: SpaceFieldsFilter | None,
        user_fields: TweetUserFieldsFilter | None,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            params = {"query": query, "max_results": max_results, "state": state.value}

            params = (
                SpaceExpansionsBuilder(params)
                .add_expansions(expansions)
                .add_space_fields(space_fields)
                .add_user_fields(user_fields)
                .build()
            )

            response = cast(Response, client.search_spaces(**params))

            meta = {}
            next_token = ""
            if response.meta:
                meta = response.meta
                if "next_token" in meta:
                    next_token = meta["next_token"]

            included = IncludesSerializer.serialize(response.includes)
            data = ResponseDataSerializer.serialize_list(response.data)

            if response.data:
                ids = [str(space["id"]) for space in response.data if "id" in space]
                titles = [space["title"] for space in data if "title" in space]
                host_ids = [space["host_ids"] for space in data if "host_ids" in space]

                return data, included, meta, ids, titles, host_ids, next_token

            raise Exception("Spaces not found")

        except tweepy.TweepyException:
            raise

    def run(
        self,
        input_data: Input,
        *,
        credentials: TwitterCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            data, included, meta, ids, titles, host_ids, next_token = (
                self.search_spaces(
                    credentials,
                    input_data.query,
                    input_data.max_results,
                    input_data.state,
                    input_data.expansions,
                    input_data.space_fields,
                    input_data.user_fields,
                )
            )

            if ids:
                yield "ids", ids
            if titles:
                yield "titles", titles
            if host_ids:
                yield "host_ids", host_ids
            if next_token:
                yield "next_token", next_token
            if data:
                yield "data", data
            if included:
                yield "includes", included
            if meta:
                yield "meta", meta

        except Exception as e:
            yield "error", handle_tweepy_exception(e)
