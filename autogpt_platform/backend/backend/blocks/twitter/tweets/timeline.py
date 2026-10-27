from collections.abc import Callable, Iterator
from typing import Any, cast

import tweepy
from tweepy.client import Response

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockEffect,
    BlockOutput,
    BlockSchemaOutput,
)
from backend.blocks.twitter._auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    TWITTER_OAUTH_IS_CONFIGURED,
    TwitterCredentials,
    TwitterCredentialsField,
    TwitterCredentialsInput,
)
from backend.blocks.twitter._builders import (
    TweetDurationBuilder,
    TweetExpansionsBuilder,
)
from backend.blocks.twitter._serializer import (
    IncludesSerializer,
    ResponseDataSerializer,
)
from backend.blocks.twitter._types import TweetExpansionInputs, TweetTimeWindowInputs
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.model import SchemaField


class TwitterGetUserMentionsBlock(Block):
    """
    Returns Tweets where a single user is mentioned, just put that user id
    """

    class Input(TweetExpansionInputs, TweetTimeWindowInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "offline.access"]
        )

        user_id: str = SchemaField(
            description="Unique identifier of the user for whom to return Tweets mentioning the user",
            placeholder="Enter user ID",
        )

        max_results: int | None = SchemaField(
            description="Number of tweets to retrieve (5-100)",
            default=10,
            advanced=True,
        )

        pagination_token: str | None = SchemaField(
            description="Token for pagination", default="", advanced=True
        )

    class Output(BlockSchemaOutput):
        # Common Outputs that user commonly uses
        ids: list[str] = SchemaField(description="List of Tweet IDs")
        texts: list[str] = SchemaField(description="All Tweet texts")

        userIds: list[str] = SchemaField(
            description="List of user ids that mentioned the user"
        )
        userNames: list[str] = SchemaField(
            description="List of user names that mentioned the user"
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

    def __init__(self):
        super().__init__(
            id="e01c890c-a630-11ef-9e20-37da24888bd0",
            description="This block retrieves Tweets mentioning a specific user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetUserMentionsBlock.Input,
            output_schema=TwitterGetUserMentionsBlock.Output,
            disabled=not TWITTER_OAUTH_IS_CONFIGURED,
            test_input={
                "user_id": "12345",
                "credentials": TEST_CREDENTIALS_INPUT,
                "max_results": 2,
                "start_time": "2024-12-14T18:30:00.000Z",
                "end_time": "2024-12-17T18:30:00.000Z",
                "since_id": "",
                "until_id": "",
                "sort_order": None,
                "pagination_token": None,
                "expansions": None,
                "media_fields": None,
                "place_fields": None,
                "poll_fields": None,
                "tweet_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["1373001119480344583", "1372627771717869568"]),
                ("texts", ["Test mention 1", "Test mention 2"]),
                ("userIds", ["67890", "67891"]),
                ("userNames", ["testuser1", "testuser2"]),
                (
                    "data",
                    [
                        {"id": "1373001119480344583", "text": "Test mention 1"},
                        {"id": "1372627771717869568", "text": "Test mention 2"},
                    ],
                ),
            ],
            test_mock={
                "get_mentions": lambda *args, **kwargs: (
                    ["1373001119480344583", "1372627771717869568"],
                    ["Test mention 1", "Test mention 2"],
                    ["67890", "67891"],
                    ["testuser1", "testuser2"],
                    [
                        {"id": "1373001119480344583", "text": "Test mention 1"},
                        {"id": "1372627771717869568", "text": "Test mention 2"},
                    ],
                    {},
                    {},
                    None,
                )
            },
            effect=BlockEffect.READ,
        )

    @staticmethod
    def get_mentions(credentials: TwitterCredentials, input_data: Input):
        return _get_tweets(
            credentials,
            tweepy.Client.get_users_mentions,
            input_data,
            id=input_data.user_id,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: TwitterCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            for output in _tweet_outputs(self.get_mentions(credentials, input_data)):
                yield output
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetHomeTimelineBlock(Block):
    """
    Returns a collection of the most recent Tweets and Retweets posted by you and users you follow
    """

    class Input(TweetExpansionInputs, TweetTimeWindowInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "offline.access"]
        )

        max_results: int | None = SchemaField(
            description="Number of tweets to retrieve (5-100)",
            default=10,
            advanced=True,
        )

        pagination_token: str | None = SchemaField(
            description="Token for pagination", default="", advanced=True
        )

    class Output(BlockSchemaOutput):
        # Common Outputs that user commonly uses
        ids: list[str] = SchemaField(description="List of Tweet IDs")
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

    def __init__(self):
        super().__init__(
            id="d222a070-a630-11ef-a18a-3f52f76c6962",
            description="This block retrieves the authenticated user's home timeline.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetHomeTimelineBlock.Input,
            output_schema=TwitterGetHomeTimelineBlock.Output,
            disabled=not TWITTER_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "max_results": 2,
                "start_time": "2024-12-14T18:30:00.000Z",
                "end_time": "2024-12-17T18:30:00.000Z",
                "since_id": None,
                "until_id": None,
                "sort_order": None,
                "pagination_token": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["1373001119480344583", "1372627771717869568"]),
                ("texts", ["Test tweet 1", "Test tweet 2"]),
                ("userIds", ["67890", "67891"]),
                ("userNames", ["testuser1", "testuser2"]),
                (
                    "data",
                    [
                        {"id": "1373001119480344583", "text": "Test tweet 1"},
                        {"id": "1372627771717869568", "text": "Test tweet 2"},
                    ],
                ),
            ],
            test_mock={
                "get_timeline": lambda *args, **kwargs: (
                    ["1373001119480344583", "1372627771717869568"],
                    ["Test tweet 1", "Test tweet 2"],
                    ["67890", "67891"],
                    ["testuser1", "testuser2"],
                    [
                        {"id": "1373001119480344583", "text": "Test tweet 1"},
                        {"id": "1372627771717869568", "text": "Test tweet 2"},
                    ],
                    {},
                    {},
                    None,
                )
            },
            effect=BlockEffect.READ,
        )

    @staticmethod
    def get_timeline(credentials: TwitterCredentials, input_data: Input):
        return _get_tweets(
            credentials,
            tweepy.Client.get_home_timeline,
            input_data,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: TwitterCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            for output in _tweet_outputs(self.get_timeline(credentials, input_data)):
                yield output
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetUserTweetsBlock(Block):
    """
    Returns Tweets composed by a single user, specified by the requested user ID
    """

    class Input(TweetExpansionInputs, TweetTimeWindowInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "offline.access"]
        )

        user_id: str = SchemaField(
            description="Unique identifier of the Twitter account (user ID) for whom to return results",
            placeholder="Enter user ID",
        )

        max_results: int | None = SchemaField(
            description="Number of tweets to retrieve (5-100)",
            default=10,
            advanced=True,
        )

        pagination_token: str | None = SchemaField(
            description="Token for pagination", default="", advanced=True
        )

    class Output(BlockSchemaOutput):
        # Common Outputs that user commonly uses
        ids: list[str] = SchemaField(description="List of Tweet IDs")
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

    def __init__(self):
        super().__init__(
            id="c44c3ef2-a630-11ef-9ff7-eb7b5ea3a5cb",
            description="This block retrieves Tweets composed by a single user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetUserTweetsBlock.Input,
            output_schema=TwitterGetUserTweetsBlock.Output,
            disabled=not TWITTER_OAUTH_IS_CONFIGURED,
            test_input={
                "user_id": "12345",
                "credentials": TEST_CREDENTIALS_INPUT,
                "max_results": 2,
                "start_time": "2024-12-14T18:30:00.000Z",
                "end_time": "2024-12-17T18:30:00.000Z",
                "since_id": None,
                "until_id": None,
                "sort_order": None,
                "pagination_token": None,
                "expansions": None,
                "media_fields": None,
                "place_fields": None,
                "poll_fields": None,
                "tweet_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["1373001119480344583", "1372627771717869568"]),
                ("texts", ["Test tweet 1", "Test tweet 2"]),
                ("userIds", ["67890", "67891"]),
                ("userNames", ["testuser1", "testuser2"]),
                (
                    "data",
                    [
                        {"id": "1373001119480344583", "text": "Test tweet 1"},
                        {"id": "1372627771717869568", "text": "Test tweet 2"},
                    ],
                ),
            ],
            test_mock={
                "get_user_tweets": lambda *args, **kwargs: (
                    ["1373001119480344583", "1372627771717869568"],
                    ["Test tweet 1", "Test tweet 2"],
                    ["67890", "67891"],
                    ["testuser1", "testuser2"],
                    [
                        {"id": "1373001119480344583", "text": "Test tweet 1"},
                        {"id": "1372627771717869568", "text": "Test tweet 2"},
                    ],
                    {},
                    {},
                    None,
                )
            },
            effect=BlockEffect.READ,
        )

    @staticmethod
    def get_user_tweets(credentials: TwitterCredentials, input_data: Input):
        return _get_tweets(
            credentials,
            tweepy.Client.get_users_tweets,
            input_data,
            id=input_data.user_id,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: TwitterCredentials,
        **kwargs,
    ) -> BlockOutput:
        try:
            for output in _tweet_outputs(self.get_user_tweets(credentials, input_data)):
                yield output
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


def _get_tweets(
    credentials: TwitterCredentials,
    fetch: Callable[..., Any],
    input_data: "TwitterGetUserMentionsBlock.Input | TwitterGetHomeTimelineBlock.Input | TwitterGetUserTweetsBlock.Input",
    **id_param: str,
):
    client = tweepy.Client(bearer_token=credentials.access_token.get_secret_value())

    params = {
        **id_param,
        "max_results": input_data.max_results,
        "pagination_token": (
            None if input_data.pagination_token == "" else input_data.pagination_token
        ),
        "user_auth": False,
    }
    params = (
        TweetExpansionsBuilder(params)
        .add_expansions(input_data.expansions)
        .add_media_fields(input_data.media_fields)
        .add_place_fields(input_data.place_fields)
        .add_poll_fields(input_data.poll_fields)
        .add_tweet_fields(input_data.tweet_fields)
        .add_user_fields(input_data.user_fields)
        .build()
    )
    params = (
        TweetDurationBuilder(params)
        .add_start_time(input_data.start_time)
        .add_end_time(input_data.end_time)
        .add_since_id(input_data.since_id)
        .add_until_id(input_data.until_id)
        .add_sort_order(input_data.sort_order)
        .build()
    )

    response = cast(Response, fetch(client, **params))

    if not response.data and not response.meta:
        raise Exception("No tweets found")

    included = IncludesSerializer.serialize(response.includes)
    data = ResponseDataSerializer.serialize_list(response.data)
    meta = response.meta or {}
    next_token = meta.get("next_token", "")

    tweet_ids = []
    tweet_texts = []
    user_ids = []
    user_names = []

    if response.data:
        tweet_ids = [str(tweet.id) for tweet in response.data]
        tweet_texts = [tweet.text for tweet in response.data]

    if "users" in included:
        user_ids = [str(user["id"]) for user in included["users"]]
        user_names = [user["username"] for user in included["users"]]

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


def _tweet_outputs(result: tuple) -> Iterator[tuple[str, Any]]:
    ids, texts, user_ids, user_names, data, included, meta, next_token = result
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
