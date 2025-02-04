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


class TwitterRetweetBlock(Block):
    """
    Retweets a tweet on Twitter
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "tweet.write", "users.read", "offline.access"]
        )

        tweet_id: str = SchemaField(
            description="ID of the tweet to retweet",
            placeholder="Enter tweet ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the retweet was successful")
        error: str = SchemaField(description="Error message if the retweet failed")

    def __init__(self):
        super().__init__(
            id="bd7b8d3a-a630-11ef-be96-6f4aa4c3c4f4",
            description="This block retweets a tweet on Twitter.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterRetweetBlock.Input,
            output_schema=TwitterRetweetBlock.Output,
            test_input={
                "tweet_id": "1234567890",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"retweet": lambda *args, **kwargs: True},
        )

    @staticmethod
    def retweet(
        credentials: TwitterCredentials,
        tweet_id: str,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.retweet(
                tweet_id=tweet_id,
                user_auth=False,
            )

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
            success = self.retweet(
                credentials,
                input_data.tweet_id,
            )
            yield "success", success
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterRemoveRetweetBlock(Block):
    """
    Removes a retweet on Twitter
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "tweet.write", "users.read", "offline.access"]
        )

        tweet_id: str = SchemaField(
            description="ID of the tweet to remove retweet",
            placeholder="Enter tweet ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the retweet was successfully removed"
        )
        error: str = SchemaField(description="Error message if the removal failed")

    def __init__(self):
        super().__init__(
            id="b6e663f0-a630-11ef-a7f0-8b9b0c542ff8",
            description="This block removes a retweet on Twitter.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterRemoveRetweetBlock.Input,
            output_schema=TwitterRemoveRetweetBlock.Output,
            test_input={
                "tweet_id": "1234567890",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"remove_retweet": lambda *args, **kwargs: True},
        )

    @staticmethod
    def remove_retweet(
        credentials: TwitterCredentials,
        tweet_id: str,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.unretweet(
                source_tweet_id=tweet_id,
                user_auth=False,
            )

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
            success = self.remove_retweet(
                credentials,
                input_data.tweet_id,
            )
            yield "success", success
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetRetweetersBlock(Block):
    """
    Gets information about who has retweeted a tweet
    """

    class Input(UserExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "offline.access"]
        )

        tweet_id: str = SchemaField(
            description="ID of the tweet to get retweeters for",
            placeholder="Enter tweet ID",
        )

        max_results: int | None = SchemaField(
            description="Maximum number of results per page (1-100)",
            default=10,
            placeholder="Enter max results",
            advanced=True,
        )

        pagination_token: str | None = SchemaField(
            description="Token for pagination",
            placeholder="Enter pagination token",
            default="",
        )

    class Output(BlockSchema):
        # Common Outputs that user commonly uses
        ids: list = SchemaField(description="List of user ids who retweeted")
        names: list = SchemaField(description="List of user names who retweeted")
        usernames: list = SchemaField(
            description="List of user usernames who retweeted"
        )
        next_token: str = SchemaField(description="Token for next page of results")

        # Complete Outputs for advanced use
        data: list[dict] = SchemaField(description="Complete Tweet data")
        included: dict = SchemaField(
            description="Additional data that you have requested (Optional) via Expansions field"
        )
        meta: dict = SchemaField(
            description="Provides metadata such as pagination info (next_token) or result counts"
        )

        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="ad7aa6fa-a630-11ef-a6b0-e7ca640aa030",
            description="This block gets information about who has retweeted a tweet.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetRetweetersBlock.Input,
            output_schema=TwitterGetRetweetersBlock.Output,
            test_input={
                "tweet_id": "1234567890",
                "credentials": TEST_CREDENTIALS_INPUT,
                "max_results": 1,
                "pagination_token": "",
                "expansions": None,
                "media_fields": None,
                "place_fields": None,
                "poll_fields": None,
                "tweet_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["12345"]),
                ("names", ["Test User"]),
                ("usernames", ["testuser"]),
                (
                    "data",
                    [{"id": "12345", "name": "Test User", "username": "testuser"}],
                ),
            ],
            test_mock={
                "get_retweeters": lambda *args, **kwargs: (
                    [{"id": "12345", "name": "Test User", "username": "testuser"}],
                    {},
                    {},
                    ["12345"],
                    ["Test User"],
                    ["testuser"],
                    None,
                )
            },
        )

    @staticmethod
    def get_retweeters(
        credentials: TwitterCredentials,
        tweet_id: str,
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
                "id": tweet_id,
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

            response = cast(Response, client.get_retweeters(**params))

            meta = {}
            ids = []
            names = []
            usernames = []
            next_token = None

            if response.meta:
                meta = response.meta
                next_token = meta.get("next_token")

            included = IncludesSerializer.serialize(response.includes)
            data = ResponseDataSerializer.serialize_list(response.data)

            if response.data:
                ids = [str(user.id) for user in response.data]
                names = [user.name for user in response.data]
                usernames = [user.username for user in response.data]
                return data, included, meta, ids, names, usernames, next_token

            raise Exception("No retweeters found")

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
            data, included, meta, ids, names, usernames, next_token = (
                self.get_retweeters(
                    credentials,
                    input_data.tweet_id,
                    input_data.max_results,
                    input_data.pagination_token,
                    input_data.expansions,
                    input_data.tweet_fields,
                    input_data.user_fields,
                )
            )

            if ids:
                yield "ids", ids
            if names:
                yield "names", names
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
