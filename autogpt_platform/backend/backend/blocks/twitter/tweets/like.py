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
    TweetExpansionsBuilder,
    UserExpansionsBuilder,
)
from backend.blocks.twitter._serializer import (
    IncludesSerializer,
    ResponseDataSerializer,
)
from backend.blocks.twitter._types import (
    ExpansionFilter,
    TweetExpansionInputs,
    TweetFieldsFilter,
    TweetMediaFieldsFilter,
    TweetPlaceFieldsFilter,
    TweetPollFieldsFilter,
    TweetUserFieldsFilter,
    UserExpansionInputs,
    UserExpansionsFilter,
)
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TwitterLikeTweetBlock(Block):
    """
    Likes a tweet
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "like.write", "users.read", "offline.access"]
        )

        tweet_id: str = SchemaField(
            description="ID of the tweet to like",
            placeholder="Enter tweet ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the operation was successful")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="4d0b4c5c-a630-11ef-8e08-1b14c507b347",
            description="This block likes a tweet.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterLikeTweetBlock.Input,
            output_schema=TwitterLikeTweetBlock.Output,
            test_input={
                "tweet_id": "1234567890",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"like_tweet": lambda *args, **kwargs: True},
        )

    @staticmethod
    def like_tweet(
        credentials: TwitterCredentials,
        tweet_id: str,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.like(tweet_id=tweet_id, user_auth=False)

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
            success = self.like_tweet(
                credentials,
                input_data.tweet_id,
            )
            yield "success", success
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetLikingUsersBlock(Block):
    """
    Gets information about users who liked a one of your tweet
    """

    class Input(UserExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "like.read", "offline.access"]
        )

        tweet_id: str = SchemaField(
            description="ID of the tweet to get liking users for",
            placeholder="Enter tweet ID",
        )
        max_results: int | None = SchemaField(
            description="Maximum number of results to return (1-100)",
            placeholder="Enter max results",
            default=10,
            advanced=True,
        )
        pagination_token: str | None = SchemaField(
            description="Token for getting next/previous page of results",
            placeholder="Enter pagination token",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        # Common Outputs that user commonly uses
        id: list[str] = SchemaField(description="All User IDs who liked the tweet")
        username: list[str] = SchemaField(
            description="All User usernames who liked the tweet"
        )
        next_token: str = SchemaField(description="Next token for pagination")

        # Complete Outputs for advanced use
        data: list[dict] = SchemaField(description="Complete Tweet data")
        included: dict = SchemaField(
            description="Additional data that you have requested (Optional) via Expansions field"
        )
        meta: dict = SchemaField(
            description="Provides metadata such as pagination info (next_token) or result counts"
        )

        # error
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="34275000-a630-11ef-b01e-5f00d9077c08",
            description="This block gets information about users who liked a tweet.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetLikingUsersBlock.Input,
            output_schema=TwitterGetLikingUsersBlock.Output,
            test_input={
                "tweet_id": "1234567890",
                "max_results": 1,
                "pagination_token": None,
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": None,
                "tweet_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("id", ["1234567890"]),
                ("username", ["testuser"]),
                ("data", [{"id": "1234567890", "username": "testuser"}]),
            ],
            test_mock={
                "get_liking_users": lambda *args, **kwargs: (
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
    def get_liking_users(
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

            response = cast(Response, client.get_liking_users(**params))

            if not response.data and not response.meta:
                raise Exception("No liking users found")

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
                user_ids = [str(user.id) for user in response.data]
                usernames = [user.username for user in response.data]

                return user_ids, usernames, data, included, meta, next_token

            raise Exception("No liking users found")

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
            ids, usernames, data, included, meta, next_token = self.get_liking_users(
                credentials,
                input_data.tweet_id,
                input_data.max_results,
                input_data.pagination_token,
                input_data.expansions,
                input_data.tweet_fields,
                input_data.user_fields,
            )
            if ids:
                yield "id", ids
            if usernames:
                yield "username", usernames
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


class TwitterGetLikedTweetsBlock(Block):
    """
    Gets information about tweets liked by you
    """

    class Input(TweetExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "like.read", "offline.access"]
        )

        user_id: str = SchemaField(
            description="ID of the user to get liked tweets for",
            placeholder="Enter user ID",
        )
        max_results: int | None = SchemaField(
            description="Maximum number of results to return (5-100)",
            placeholder="100",
            default=10,
            advanced=True,
        )
        pagination_token: str | None = SchemaField(
            description="Token for getting next/previous page of results",
            placeholder="Enter pagination token",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        # Common Outputs that user commonly uses
        ids: list[str] = SchemaField(description="All Tweet IDs")
        texts: list[str] = SchemaField(description="All Tweet texts")
        userIds: list[str] = SchemaField(
            description="List of user ids that authored the tweets"
        )
        userNames: list[str] = SchemaField(
            description="List of user names that authored the tweets"
        )
        next_token: str = SchemaField(description="Next token for pagination")

        # Complete Outputs for advanced use
        data: list[dict] = SchemaField(description="Complete Tweet data")
        included: dict = SchemaField(
            description="Additional data that you have requested (Optional) via Expansions field"
        )
        meta: dict = SchemaField(
            description="Provides metadata such as pagination info (next_token) or result counts"
        )

        # error
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="292e7c78-a630-11ef-9f40-df5dffaca106",
            description="This block gets information about tweets liked by a user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetLikedTweetsBlock.Input,
            output_schema=TwitterGetLikedTweetsBlock.Output,
            test_input={
                "user_id": "1234567890",
                "max_results": 2,
                "pagination_token": None,
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": None,
                "media_fields": None,
                "place_fields": None,
                "poll_fields": None,
                "tweet_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["12345", "67890"]),
                ("texts", ["Tweet 1", "Tweet 2"]),
                ("userIds", ["67890", "67891"]),
                ("userNames", ["testuser1", "testuser2"]),
                (
                    "data",
                    [
                        {"id": "12345", "text": "Tweet 1"},
                        {"id": "67890", "text": "Tweet 2"},
                    ],
                ),
            ],
            test_mock={
                "get_liked_tweets": lambda *args, **kwargs: (
                    ["12345", "67890"],
                    ["Tweet 1", "Tweet 2"],
                    ["67890", "67891"],
                    ["testuser1", "testuser2"],
                    [
                        {"id": "12345", "text": "Tweet 1"},
                        {"id": "67890", "text": "Tweet 2"},
                    ],
                    {},
                    {},
                    None,
                )
            },
        )

    @staticmethod
    def get_liked_tweets(
        credentials: TwitterCredentials,
        user_id: str,
        max_results: int | None,
        pagination_token: str | None,
        expansions: ExpansionFilter | None,
        media_fields: TweetMediaFieldsFilter | None,
        place_fields: TweetPlaceFieldsFilter | None,
        poll_fields: TweetPollFieldsFilter | None,
        tweet_fields: TweetFieldsFilter | None,
        user_fields: TweetUserFieldsFilter | None,
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
                TweetExpansionsBuilder(params)
                .add_expansions(expansions)
                .add_media_fields(media_fields)
                .add_place_fields(place_fields)
                .add_poll_fields(poll_fields)
                .add_tweet_fields(tweet_fields)
                .add_user_fields(user_fields)
                .build()
            )

            response = cast(Response, client.get_liked_tweets(**params))

            if not response.data and not response.meta:
                raise Exception("No liked tweets found")

            meta = {}
            tweet_ids = []
            tweet_texts = []
            user_ids = []
            user_names = []
            next_token = None

            if response.meta:
                meta = response.meta
                next_token = meta.get("next_token")

            included = IncludesSerializer.serialize(response.includes)
            data = ResponseDataSerializer.serialize_list(response.data)

            if response.data:
                tweet_ids = [str(tweet.id) for tweet in response.data]
                tweet_texts = [tweet.text for tweet in response.data]

                if "users" in response.includes:
                    user_ids = [str(user["id"]) for user in response.includes["users"]]
                    user_names = [
                        user["username"] for user in response.includes["users"]
                    ]

                return (
                    tweet_ids,
                    tweet_texts,
                    user_ids,
                    user_names,
                    data,
                    included,
                    meta,
                    next_token,
                )

            raise Exception("No liked tweets found")

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
            ids, texts, user_ids, user_names, data, included, meta, next_token = (
                self.get_liked_tweets(
                    credentials,
                    input_data.user_id,
                    input_data.max_results,
                    input_data.pagination_token,
                    input_data.expansions,
                    input_data.media_fields,
                    input_data.place_fields,
                    input_data.poll_fields,
                    input_data.tweet_fields,
                    input_data.user_fields,
                )
            )
            if ids:
                yield "ids", ids
            if texts:
                yield "texts", texts
            if user_ids:
                yield "userIds", user_ids
            if user_names:
                yield "userNames", user_names
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


class TwitterUnlikeTweetBlock(Block):
    """
    Unlikes a tweet that was previously liked
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "like.write", "users.read", "offline.access"]
        )

        tweet_id: str = SchemaField(
            description="ID of the tweet to unlike",
            placeholder="Enter tweet ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the operation was successful")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="1ed5eab8-a630-11ef-8e21-cbbbc80cbb85",
            description="This block unlikes a tweet.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterUnlikeTweetBlock.Input,
            output_schema=TwitterUnlikeTweetBlock.Output,
            test_input={
                "tweet_id": "1234567890",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"unlike_tweet": lambda *args, **kwargs: True},
        )

    @staticmethod
    def unlike_tweet(
        credentials: TwitterCredentials,
        tweet_id: str,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.unlike(tweet_id=tweet_id, user_auth=False)

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
            success = self.unlike_tweet(
                credentials,
                input_data.tweet_id,
            )
            yield "success", success
        except Exception as e:
            yield "error", handle_tweepy_exception(e)
