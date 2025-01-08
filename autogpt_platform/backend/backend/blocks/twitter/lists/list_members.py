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
from backend.blocks.twitter._builders import (
    ListExpansionsBuilder,
    UserExpansionsBuilder,
)
from backend.blocks.twitter._serializer import (
    IncludesSerializer,
    ResponseDataSerializer,
)
from backend.blocks.twitter._types import (
    ListExpansionInputs,
    ListExpansionsFilter,
    ListFieldsFilter,
    TweetFieldsFilter,
    TweetUserFieldsFilter,
    UserExpansionInputs,
    UserExpansionsFilter,
)
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TwitterRemoveListMemberBlock(Block):
    """
    Removes a member from a Twitter List that the authenticated user owns
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["list.write", "users.read", "tweet.read", "offline.access"]
        )

        list_id: str = SchemaField(
            description="The ID of the List to remove the member from",
            placeholder="Enter list ID",
            required=True,
        )

        user_id: str = SchemaField(
            description="The ID of the user to remove from the List",
            placeholder="Enter user ID to remove",
            required=True,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the member was successfully removed"
        )
        error: str = SchemaField(description="Error message if the removal failed")

    def __init__(self):
        super().__init__(
            id="5a3d1320-a62f-11ef-b7ce-a79e7656bcb0",
            description="This block removes a specified user from a Twitter List owned by the authenticated user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterRemoveListMemberBlock.Input,
            output_schema=TwitterRemoveListMemberBlock.Output,
            test_input={
                "list_id": "123456789",
                "user_id": "987654321",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"remove_list_member": lambda *args, **kwargs: True},
        )

    @staticmethod
    def remove_list_member(credentials: TwitterCredentials, list_id: str, user_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )
            client.remove_list_member(id=list_id, user_id=user_id, user_auth=False)
            return True
        except tweepy.TweepyException:
            raise
        except Exception:
            raise

    def run(
        self,
        input_data: Input,
        *,
        credentials: TwitterCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            success = self.remove_list_member(
                credentials, input_data.list_id, input_data.user_id
            )
            yield "success", success

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterAddListMemberBlock(Block):
    """
    Adds a member to a Twitter List that the authenticated user owns
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["list.write", "users.read", "tweet.read", "offline.access"]
        )

        list_id: str = SchemaField(
            description="The ID of the List to add the member to",
            placeholder="Enter list ID",
            required=True,
        )

        user_id: str = SchemaField(
            description="The ID of the user to add to the List",
            placeholder="Enter user ID to add",
            required=True,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the member was successfully added"
        )
        error: str = SchemaField(description="Error message if the addition failed")

    def __init__(self):
        super().__init__(
            id="3ee8284e-a62f-11ef-84e4-8f6e2cbf0ddb",
            description="This block adds a specified user to a Twitter List owned by the authenticated user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterAddListMemberBlock.Input,
            output_schema=TwitterAddListMemberBlock.Output,
            test_input={
                "list_id": "123456789",
                "user_id": "987654321",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"add_list_member": lambda *args, **kwargs: True},
        )

    @staticmethod
    def add_list_member(credentials: TwitterCredentials, list_id: str, user_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )
            client.add_list_member(id=list_id, user_id=user_id, user_auth=False)
            return True
        except tweepy.TweepyException:
            raise
        except Exception:
            raise

    def run(
        self,
        input_data: Input,
        *,
        credentials: TwitterCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            success = self.add_list_member(
                credentials, input_data.list_id, input_data.user_id
            )
            yield "success", success

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetListMembersBlock(Block):
    """
    Gets the members of a specified Twitter List
    """

    class Input(UserExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["list.read", "offline.access"]
        )

        list_id: str = SchemaField(
            description="The ID of the List to get members from",
            placeholder="Enter list ID",
            required=True,
        )

        max_results: int | None = SchemaField(
            description="Maximum number of results per page (1-100)",
            placeholder="Enter max results",
            default=10,
            advanced=True,
        )

        pagination_token: str | None = SchemaField(
            description="Token for pagination of results",
            placeholder="Enter pagination token",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        ids: list[str] = SchemaField(description="List of member user IDs")
        usernames: list[str] = SchemaField(description="List of member usernames")
        next_token: str = SchemaField(description="Next token for pagination")

        data: list[dict] = SchemaField(
            description="Complete user data for list members"
        )
        included: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        meta: dict = SchemaField(description="Metadata including pagination info")

        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="4dba046e-a62f-11ef-b69a-87240c84b4c7",
            description="This block retrieves the members of a specified Twitter List.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetListMembersBlock.Input,
            output_schema=TwitterGetListMembersBlock.Output,
            test_input={
                "list_id": "123456789",
                "max_results": 2,
                "pagination_token": None,
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": None,
                "tweet_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["12345", "67890"]),
                ("usernames", ["testuser1", "testuser2"]),
                (
                    "data",
                    [
                        {"id": "12345", "username": "testuser1"},
                        {"id": "67890", "username": "testuser2"},
                    ],
                ),
            ],
            test_mock={
                "get_list_members": lambda *args, **kwargs: (
                    ["12345", "67890"],
                    ["testuser1", "testuser2"],
                    [
                        {"id": "12345", "username": "testuser1"},
                        {"id": "67890", "username": "testuser2"},
                    ],
                    {},
                    {},
                    None,
                )
            },
        )

    @staticmethod
    def get_list_members(
        credentials: TwitterCredentials,
        list_id: str,
        max_results: int | None,
        pagination_token: str | None,
        expansions: UserExpansionsFilter | None,
        tweet_fields: TweetFieldsFilter | None,
        user_fields: TweetUserFieldsFilter | None,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            params = {
                "id": list_id,
                "max_results": max_results,
                "pagination_token": (
                    None if pagination_token == "" else pagination_token
                ),
                "user_auth": False,
            }

            params = (
                UserExpansionsBuilder(params)
                .add_expansions(expansions)
                .add_tweet_fields(tweet_fields)
                .add_user_fields(user_fields)
                .build()
            )

            response = cast(Response, client.get_list_members(**params))

            meta = {}
            included = {}
            next_token = None
            user_ids = []
            usernames = []

            if response.meta:
                meta = response.meta
                next_token = meta.get("next_token")

            if response.includes:
                included = IncludesSerializer.serialize(response.includes)

            if response.data:
                data = ResponseDataSerializer.serialize_list(response.data)
                user_ids = [str(user.id) for user in response.data]
                usernames = [user.username for user in response.data]
                return user_ids, usernames, data, included, meta, next_token

            raise Exception("List members not found")

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
            ids, usernames, data, included, meta, next_token = self.get_list_members(
                credentials,
                input_data.list_id,
                input_data.max_results,
                input_data.pagination_token,
                input_data.expansions,
                input_data.tweet_fields,
                input_data.user_fields,
            )

            if ids:
                yield "ids", ids
            if usernames:
                yield "usernames", usernames
            if next_token:
                yield "next_token", next_token
            if data:
                yield "data", data
            if included:
                yield "included", included
            if meta:
                yield "meta", meta

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetListMembershipsBlock(Block):
    """
    Gets all Lists that a specified user is a member of
    """

    class Input(ListExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["list.read", "offline.access"]
        )

        user_id: str = SchemaField(
            description="The ID of the user whose List memberships to retrieve",
            placeholder="Enter user ID",
            required=True,
        )

        max_results: int | None = SchemaField(
            description="Maximum number of results per page (1-100)",
            placeholder="Enter max results",
            advanced=True,
            default=10,
        )

        pagination_token: str | None = SchemaField(
            description="Token for pagination of results",
            placeholder="Enter pagination token",
            advanced=True,
            default="",
        )

    class Output(BlockSchema):
        list_ids: list[str] = SchemaField(description="List of list IDs")
        next_token: str = SchemaField(description="Next token for pagination")

        data: list[dict] = SchemaField(description="List membership data")
        included: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        meta: dict = SchemaField(description="Metadata about pagination")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="46e6429c-a62f-11ef-81c0-2b55bc7823ba",
            description="This block retrieves all Lists that a specified user is a member of.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetListMembershipsBlock.Input,
            output_schema=TwitterGetListMembershipsBlock.Output,
            test_input={
                "user_id": "123456789",
                "max_results": 1,
                "pagination_token": None,
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": None,
                "list_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("list_ids", ["84839422"]),
                ("data", [{"id": "84839422"}]),
            ],
            test_mock={
                "get_list_memberships": lambda *args, **kwargs: (
                    [{"id": "84839422"}],
                    {},
                    {},
                    ["84839422"],
                    None,
                )
            },
        )

    @staticmethod
    def get_list_memberships(
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

            response = cast(Response, client.get_list_memberships(**params))

            meta = {}
            included = {}
            next_token = None
            list_ids = []

            if response.meta:
                meta = response.meta
                next_token = meta.get("next_token")

            if response.includes:
                included = IncludesSerializer.serialize(response.includes)

            if response.data:
                data = ResponseDataSerializer.serialize_list(response.data)
                list_ids = [str(lst.id) for lst in response.data]
                return data, included, meta, list_ids, next_token

            raise Exception("List memberships not found")

        except tweepy.TweepyException:
            raise
        except Exception:
            raise

    def run(
        self,
        input_data: Input,
        *,
        credentials: TwitterCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            data, included, meta, list_ids, next_token = self.get_list_memberships(
                credentials,
                input_data.user_id,
                input_data.max_results,
                input_data.pagination_token,
                input_data.expansions,
                input_data.user_fields,
                input_data.list_fields,
            )

            if list_ids:
                yield "list_ids", list_ids
            if next_token:
                yield "next_token", next_token
            if data:
                yield "data", data
            if included:
                yield "included", included
            if meta:
                yield "meta", meta

        except Exception as e:
            yield "error", handle_tweepy_exception(e)
