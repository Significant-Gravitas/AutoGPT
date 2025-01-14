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


class TwitterUnfollowUserBlock(Block):
    """
    Allows a user to unfollow another user specified by target user ID.
    The request succeeds with no action when the authenticated user sends a request to a user they're not following or have already unfollowed.
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["users.read", "users.write", "follows.write", "offline.access"]
        )

        target_user_id: str = SchemaField(
            description="The user ID of the user that you would like to unfollow",
            placeholder="Enter target user ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the unfollow action was successful"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="37e386a4-a631-11ef-b7bd-b78204b35fa4",
            description="This block unfollows a specified Twitter user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterUnfollowUserBlock.Input,
            output_schema=TwitterUnfollowUserBlock.Output,
            test_input={
                "target_user_id": "12345",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"unfollow_user": lambda *args, **kwargs: True},
        )

    @staticmethod
    def unfollow_user(credentials: TwitterCredentials, target_user_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.unfollow_user(target_user_id=target_user_id, user_auth=False)

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
            success = self.unfollow_user(credentials, input_data.target_user_id)
            yield "success", success

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterFollowUserBlock(Block):
    """
    Allows a user to follow another user specified by target user ID. If the target user does not have public Tweets,
    this endpoint will send a follow request. The request succeeds with no action when the authenticated user sends a
    request to a user they're already following, or if they're sending a follower request to a user that does not have
    public Tweets.
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["users.read", "users.write", "follows.write", "offline.access"]
        )

        target_user_id: str = SchemaField(
            description="The user ID of the user that you would like to follow",
            placeholder="Enter target user ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the follow action was successful"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="1aae6a5e-a631-11ef-a090-435900c6d429",
            description="This block follows a specified Twitter user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterFollowUserBlock.Input,
            output_schema=TwitterFollowUserBlock.Output,
            test_input={
                "target_user_id": "12345",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"follow_user": lambda *args, **kwargs: True},
        )

    @staticmethod
    def follow_user(credentials: TwitterCredentials, target_user_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.follow_user(target_user_id=target_user_id, user_auth=False)

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
            success = self.follow_user(credentials, input_data.target_user_id)
            yield "success", success

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetFollowersBlock(Block):
    """
    Retrieves a list of followers for a specified Twitter user ID
    """

    class Input(UserExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["users.read", "offline.access", "follows.read"]
        )

        target_user_id: str = SchemaField(
            description="The user ID whose followers you would like to retrieve",
            placeholder="Enter target user ID",
        )

        max_results: int | None = SchemaField(
            description="Maximum number of results to return (1-1000, default 100)",
            placeholder="Enter max results",
            default=10,
            advanced=True,
        )

        pagination_token: str | None = SchemaField(
            description="Token for retrieving next/previous page of results",
            placeholder="Enter pagination token",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        ids: list[str] = SchemaField(description="List of follower user IDs")
        usernames: list[str] = SchemaField(description="List of follower usernames")
        next_token: str = SchemaField(description="Next token for pagination")

        data: list[dict] = SchemaField(description="Complete user data for followers")
        includes: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        meta: dict = SchemaField(description="Metadata including pagination info")

        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="30f66410-a631-11ef-8fe7-d7f888b4f43c",
            description="This block retrieves followers of a specified Twitter user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetFollowersBlock.Input,
            output_schema=TwitterGetFollowersBlock.Output,
            test_input={
                "target_user_id": "12345",
                "max_results": 1,
                "pagination_token": "",
                "expansions": None,
                "tweet_fields": None,
                "user_fields": None,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["1234567890"]),
                ("usernames", ["testuser"]),
                ("data", [{"id": "1234567890", "username": "testuser"}]),
            ],
            test_mock={
                "get_followers": lambda *args, **kwargs: (
                    ["1234567890"],
                    ["testuser"],
                    [{"id": "1234567890", "username": "testuser"}],
                    {},
                    {},
                    None,
                )
            },
        )

    @staticmethod
    def get_followers(
        credentials: TwitterCredentials,
        target_user_id: str,
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
                "id": target_user_id,
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

            response = cast(Response, client.get_users_followers(**params))

            meta = {}
            follower_ids = []
            follower_usernames = []
            next_token = None

            if response.meta:
                meta = response.meta
                next_token = meta.get("next_token")

            included = IncludesSerializer.serialize(response.includes)
            data = ResponseDataSerializer.serialize_list(response.data)

            if response.data:
                follower_ids = [str(user.id) for user in response.data]
                follower_usernames = [user.username for user in response.data]

                return (
                    follower_ids,
                    follower_usernames,
                    data,
                    included,
                    meta,
                    next_token,
                )

            raise Exception("Followers not found")

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
            ids, usernames, data, includes, meta, next_token = self.get_followers(
                credentials,
                input_data.target_user_id,
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


class TwitterGetFollowingBlock(Block):
    """
    Retrieves a list of users that a specified Twitter user ID is following
    """

    class Input(UserExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["users.read", "offline.access", "follows.read"]
        )

        target_user_id: str = SchemaField(
            description="The user ID whose following you would like to retrieve",
            placeholder="Enter target user ID",
        )

        max_results: int | None = SchemaField(
            description="Maximum number of results to return (1-1000, default 100)",
            placeholder="Enter max results",
            default=10,
            advanced=True,
        )

        pagination_token: str | None = SchemaField(
            description="Token for retrieving next/previous page of results",
            placeholder="Enter pagination token",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        ids: list[str] = SchemaField(description="List of following user IDs")
        usernames: list[str] = SchemaField(description="List of following usernames")
        next_token: str = SchemaField(description="Next token for pagination")

        data: list[dict] = SchemaField(description="Complete user data for following")
        includes: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        meta: dict = SchemaField(description="Metadata including pagination info")

        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="264a399c-a631-11ef-a97d-bfde4ca91173",
            description="This block retrieves the users that a specified Twitter user is following.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetFollowingBlock.Input,
            output_schema=TwitterGetFollowingBlock.Output,
            test_input={
                "target_user_id": "12345",
                "max_results": 1,
                "pagination_token": None,
                "expansions": None,
                "tweet_fields": None,
                "user_fields": None,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["1234567890"]),
                ("usernames", ["testuser"]),
                ("data", [{"id": "1234567890", "username": "testuser"}]),
            ],
            test_mock={
                "get_following": lambda *args, **kwargs: (
                    ["1234567890"],
                    ["testuser"],
                    [{"id": "1234567890", "username": "testuser"}],
                    {},
                    {},
                    None,
                )
            },
        )

    @staticmethod
    def get_following(
        credentials: TwitterCredentials,
        target_user_id: str,
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
                "id": target_user_id,
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

            response = cast(Response, client.get_users_following(**params))

            meta = {}
            following_ids = []
            following_usernames = []
            next_token = None

            if response.meta:
                meta = response.meta
                next_token = meta.get("next_token")

            included = IncludesSerializer.serialize(response.includes)
            data = ResponseDataSerializer.serialize_list(response.data)

            if response.data:
                following_ids = [str(user.id) for user in response.data]
                following_usernames = [user.username for user in response.data]

                return (
                    following_ids,
                    following_usernames,
                    data,
                    included,
                    meta,
                    next_token,
                )

            raise Exception("Following not found")

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
            ids, usernames, data, includes, meta, next_token = self.get_following(
                credentials,
                input_data.target_user_id,
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
