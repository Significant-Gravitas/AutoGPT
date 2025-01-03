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
from backend.blocks.twitter._builders import ListExpansionsBuilder
from backend.blocks.twitter._serializer import (
    IncludesSerializer,
    ResponseDataSerializer,
)
from backend.blocks.twitter._types import (
    ListExpansionInputs,
    ListExpansionsFilter,
    ListFieldsFilter,
    TweetUserFieldsFilter,
)
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TwitterGetListBlock(Block):
    """
    Gets information about a Twitter List specified by ID
    """

    class Input(ListExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "offline.access"]
        )

        list_id: str = SchemaField(
            description="The ID of the List to lookup",
            placeholder="Enter list ID",
            required=True,
        )

    class Output(BlockSchema):
        # Common outputs
        id: str = SchemaField(description="ID of the Twitter List")
        name: str = SchemaField(description="Name of the Twitter List")
        owner_id: str = SchemaField(description="ID of the List owner")
        owner_username: str = SchemaField(description="Username of the List owner")

        # Complete outputs
        data: dict = SchemaField(description="Complete list data")
        included: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        meta: dict = SchemaField(description="Metadata about the response")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="34ebc80a-a62f-11ef-9c2a-3fcab6c07079",
            description="This block retrieves information about a specified Twitter List.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetListBlock.Input,
            output_schema=TwitterGetListBlock.Output,
            test_input={
                "list_id": "84839422",
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": None,
                "list_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("id", "84839422"),
                ("name", "Official Twitter Accounts"),
                ("owner_id", "2244994945"),
                ("owner_username", "TwitterAPI"),
                ("data", {"id": "84839422", "name": "Official Twitter Accounts"}),
            ],
            test_mock={
                "get_list": lambda *args, **kwargs: (
                    {"id": "84839422", "name": "Official Twitter Accounts"},
                    {},
                    {},
                    "2244994945",
                    "TwitterAPI",
                )
            },
        )

    @staticmethod
    def get_list(
        credentials: TwitterCredentials,
        list_id: str,
        expansions: ListExpansionsFilter | None,
        user_fields: TweetUserFieldsFilter | None,
        list_fields: ListFieldsFilter | None,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            params = {"id": list_id, "user_auth": False}

            params = (
                ListExpansionsBuilder(params)
                .add_expansions(expansions)
                .add_user_fields(user_fields)
                .add_list_fields(list_fields)
                .build()
            )

            response = cast(Response, client.get_list(**params))

            meta = {}
            owner_id = ""
            owner_username = ""
            included = {}

            if response.includes:
                included = IncludesSerializer.serialize(response.includes)

            if "users" in included:
                owner_id = str(included["users"][0]["id"])
                owner_username = included["users"][0]["username"]

            if response.meta:
                meta = response.meta

            if response.data:
                data_dict = ResponseDataSerializer.serialize_dict(response.data)
                return data_dict, included, meta, owner_id, owner_username

            raise Exception("List not found")

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
            list_data, included, meta, owner_id, owner_username = self.get_list(
                credentials,
                input_data.list_id,
                input_data.expansions,
                input_data.user_fields,
                input_data.list_fields,
            )

            yield "id", str(list_data["id"])
            yield "name", list_data["name"]
            if owner_id:
                yield "owner_id", owner_id
            if owner_username:
                yield "owner_username", owner_username
            yield "data", {"id": list_data["id"], "name": list_data["name"]}
            if included:
                yield "included", included
            if meta:
                yield "meta", meta

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetOwnedListsBlock(Block):
    """
    Gets all Lists owned by the specified user
    """

    class Input(ListExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "list.read", "offline.access"]
        )

        user_id: str = SchemaField(
            description="The user ID whose owned Lists to retrieve",
            placeholder="Enter user ID",
            required=True,
        )

        max_results: int | None = SchemaField(
            description="Maximum number of results per page (1-100)",
            placeholder="Enter max results (default 100)",
            advanced=True,
            default=10,
        )

        pagination_token: str | None = SchemaField(
            description="Token for pagination",
            placeholder="Enter pagination token",
            advanced=True,
            default="",
        )

    class Output(BlockSchema):
        # Common outputs
        list_ids: list[str] = SchemaField(description="List ids of the owned lists")
        list_names: list[str] = SchemaField(description="List names of the owned lists")
        next_token: str = SchemaField(description="Token for next page of results")

        # Complete outputs
        data: list[dict] = SchemaField(description="Complete owned lists data")
        included: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        meta: dict = SchemaField(description="Metadata about the response")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="2b6bdb26-a62f-11ef-a9ce-ff89c2568726",
            description="This block retrieves all Lists owned by a specified Twitter user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetOwnedListsBlock.Input,
            output_schema=TwitterGetOwnedListsBlock.Output,
            test_input={
                "user_id": "2244994945",
                "max_results": 10,
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": None,
                "list_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("list_ids", ["84839422"]),
                ("list_names", ["Official Twitter Accounts"]),
                ("data", [{"id": "84839422", "name": "Official Twitter Accounts"}]),
            ],
            test_mock={
                "get_owned_lists": lambda *args, **kwargs: (
                    [{"id": "84839422", "name": "Official Twitter Accounts"}],
                    {},
                    {},
                    ["84839422"],
                    ["Official Twitter Accounts"],
                    None,
                )
            },
        )

    @staticmethod
    def get_owned_lists(
        credentials: TwitterCredentials,
        user_id: str,
        max_results: int | None,
        pagination_token: str | None,
        expansions: ListExpansionsFilter | None,
        user_fields: TweetUserFieldsFilter | None,
        list_fields: ListFieldsFilter | None,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            params = {
                "id": user_id,
                "max_results": max_results,
                "pagination_token": (
                    None if pagination_token == "" else pagination_token
                ),
                "user_auth": False,
            }

            params = (
                ListExpansionsBuilder(params)
                .add_expansions(expansions)
                .add_user_fields(user_fields)
                .add_list_fields(list_fields)
                .build()
            )

            response = cast(Response, client.get_owned_lists(**params))

            meta = {}
            included = {}
            list_ids = []
            list_names = []
            next_token = None

            if response.meta:
                meta = response.meta
                next_token = meta.get("next_token")

            if response.includes:
                included = IncludesSerializer.serialize(response.includes)

            if response.data:
                data = ResponseDataSerializer.serialize_list(response.data)
                list_ids = [
                    str(item.id) for item in response.data if hasattr(item, "id")
                ]
                list_names = [
                    item.name for item in response.data if hasattr(item, "name")
                ]

                return data, included, meta, list_ids, list_names, next_token

            raise Exception("User have no owned list")

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
            list_data, included, meta, list_ids, list_names, next_token = (
                self.get_owned_lists(
                    credentials,
                    input_data.user_id,
                    input_data.max_results,
                    input_data.pagination_token,
                    input_data.expansions,
                    input_data.user_fields,
                    input_data.list_fields,
                )
            )

            if list_ids:
                yield "list_ids", list_ids
            if list_names:
                yield "list_names", list_names
            if next_token:
                yield "next_token", next_token
            if list_data:
                yield "data", list_data
            if included:
                yield "included", included
            if meta:
                yield "meta", meta

        except Exception as e:
            yield "error", handle_tweepy_exception(e)
