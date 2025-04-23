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
from backend.blocks.twitter._builders import UserExpansionsBuilder
from backend.blocks.twitter._serializer import (
    IncludesSerializer,
    ResponseDataSerializer,
)
from backend.blocks.twitter._types import (
    TweetFieldsFilter,
    TweetUserFieldsFilter,
    UserExpansionInputs,
    UserExpansionsFilter,
)
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TwitterUnmuteUserBlock(Block):
    """
    Allows a user to unmute another user specified by target user ID.
    The request succeeds with no action when the user sends a request to a user they're not muting or have already unmuted.
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["users.read", "users.write", "offline.access"]
        )

        target_user_id: str = SchemaField(
            description="The user ID of the user that you would like to unmute",
            placeholder="Enter target user ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the unmute action was successful"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="40458504-a631-11ef-940b-eff92be55422",
            description="This block unmutes a specified Twitter user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterUnmuteUserBlock.Input,
            output_schema=TwitterUnmuteUserBlock.Output,
            test_input={
                "target_user_id": "12345",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"unmute_user": lambda *args, **kwargs: True},
        )

    @staticmethod
    def unmute_user(credentials: TwitterCredentials, target_user_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.unmute(target_user_id=target_user_id, user_auth=False)

            return True

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
            success = self.unmute_user(credentials, input_data.target_user_id)
            yield "success", success

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetMutedUsersBlock(Block):
    """
    Returns a list of users who are muted by the authenticating user
    """

    class Input(UserExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["users.read", "offline.access"]
        )

        max_results: int | None = SchemaField(
            description="The maximum number of results to be returned per page (1-1000). Default is 100.",
            placeholder="Enter max results",
            default=10,
            advanced=True,
        )

        pagination_token: str | None = SchemaField(
            description="Token to request next/previous page of results",
            placeholder="Enter pagination token",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        ids: list[str] = SchemaField(description="List of muted user IDs")
        usernames: list[str] = SchemaField(description="List of muted usernames")
        next_token: str = SchemaField(description="Next token for pagination")

        data: list[dict] = SchemaField(description="Complete user data for muted users")
        includes: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        meta: dict = SchemaField(description="Metadata including pagination info")

        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="475024da-a631-11ef-9ccd-f724b8b03cda",
            description="This block gets a list of users muted by the authenticating user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetMutedUsersBlock.Input,
            output_schema=TwitterGetMutedUsersBlock.Output,
            test_input={
                "max_results": 2,
                "pagination_token": "",
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
                "get_muted_users": lambda *args, **kwargs: (
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
    def get_muted_users(
        credentials: TwitterCredentials,
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

            response = cast(Response, client.get_muted(**params))

            meta = {}
            user_ids = []
            usernames = []
            next_token = None

            if response.meta:
                meta = response.meta
                next_token = meta.get("next_token")

            included = IncludesSerializer.serialize(response.includes)
            data = ResponseDataSerializer.serialize_list(response.data)

            if response.data:
                user_ids = [str(item.id) for item in response.data]
                usernames = [item.username for item in response.data]

                return user_ids, usernames, data, included, meta, next_token

            raise Exception("Muted users not found")

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
            ids, usernames, data, includes, meta, next_token = self.get_muted_users(
                credentials,
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
            if includes:
                yield "includes", includes
            if meta:
                yield "meta", meta
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterMuteUserBlock(Block):
    """
    Allows a user to mute another user specified by target user ID
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["users.read", "users.write", "offline.access"]
        )

        target_user_id: str = SchemaField(
            description="The user ID of the user that you would like to mute",
            placeholder="Enter target user ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the mute action was successful"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="4d1919d0-a631-11ef-90ab-3b73af9ce8f1",
            description="This block mutes a specified Twitter user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterMuteUserBlock.Input,
            output_schema=TwitterMuteUserBlock.Output,
            test_input={
                "target_user_id": "12345",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"mute_user": lambda *args, **kwargs: True},
        )

    @staticmethod
    def mute_user(credentials: TwitterCredentials, target_user_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.mute(target_user_id=target_user_id, user_auth=False)

            return True

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
            success = self.mute_user(credentials, input_data.target_user_id)
            yield "success", success
        except Exception as e:
            yield "error", handle_tweepy_exception(e)
