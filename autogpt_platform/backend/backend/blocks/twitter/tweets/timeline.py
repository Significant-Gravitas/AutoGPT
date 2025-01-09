from datetime import datetime
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
    TweetDurationBuilder,
    TweetExpansionsBuilder,
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
    TweetTimeWindowInputs,
    TweetUserFieldsFilter,
)
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
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

    class Output(BlockSchema):
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
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="e01c890c-a630-11ef-9e20-37da24888bd0",
            description="This block retrieves Tweets mentioning a specific user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetUserMentionsBlock.Input,
            output_schema=TwitterGetUserMentionsBlock.Output,
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
        )

    @staticmethod
    def get_mentions(
        credentials: TwitterCredentials,
        user_id: str,
        max_results: int | None,
        start_time: datetime | None,
        end_time: datetime | None,
        since_id: str | None,
        until_id: str | None,
        sort_order: str | None,
        pagination: str | None,
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
                "pagination_token": None if pagination == "" else pagination,
                "user_auth": False,
            }

            # Adding expansions to params If required by the user
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

            # Adding time window to params If required by the user
            params = (
                TweetDurationBuilder(params)
                .add_start_time(start_time)
                .add_end_time(end_time)
                .add_since_id(since_id)
                .add_until_id(until_id)
                .add_sort_order(sort_order)
                .build()
            )

            response = cast(
                Response,
                client.get_users_mentions(**params),
            )

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
                self.get_mentions(
                    credentials,
                    input_data.user_id,
                    input_data.max_results,
                    input_data.start_time,
                    input_data.end_time,
                    input_data.since_id,
                    input_data.until_id,
                    input_data.sort_order,
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

    class Output(BlockSchema):
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
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="d222a070-a630-11ef-a18a-3f52f76c6962",
            description="This block retrieves the authenticated user's home timeline.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetHomeTimelineBlock.Input,
            output_schema=TwitterGetHomeTimelineBlock.Output,
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
        )

    @staticmethod
    def get_timeline(
        credentials: TwitterCredentials,
        max_results: int | None,
        start_time: datetime | None,
        end_time: datetime | None,
        since_id: str | None,
        until_id: str | None,
        sort_order: str | None,
        pagination: str | None,
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
                "max_results": max_results,
                "pagination_token": None if pagination == "" else pagination,
                "user_auth": False,
            }

            # Adding expansions to params If required by the user
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

            # Adding time window to params If required by the user
            params = (
                TweetDurationBuilder(params)
                .add_start_time(start_time)
                .add_end_time(end_time)
                .add_since_id(since_id)
                .add_until_id(until_id)
                .add_sort_order(sort_order)
                .build()
            )

            response = cast(
                Response,
                client.get_home_timeline(**params),
            )

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
                self.get_timeline(
                    credentials,
                    input_data.max_results,
                    input_data.start_time,
                    input_data.end_time,
                    input_data.since_id,
                    input_data.until_id,
                    input_data.sort_order,
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

    class Output(BlockSchema):
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
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="c44c3ef2-a630-11ef-9ff7-eb7b5ea3a5cb",
            description="This block retrieves Tweets composed by a single user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetUserTweetsBlock.Input,
            output_schema=TwitterGetUserTweetsBlock.Output,
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
        )

    @staticmethod
    def get_user_tweets(
        credentials: TwitterCredentials,
        user_id: str,
        max_results: int | None,
        start_time: datetime | None,
        end_time: datetime | None,
        since_id: str | None,
        until_id: str | None,
        sort_order: str | None,
        pagination: str | None,
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
                "pagination_token": None if pagination == "" else pagination,
                "user_auth": False,
            }

            # Adding expansions to params If required by the user
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

            # Adding time window to params If required by the user
            params = (
                TweetDurationBuilder(params)
                .add_start_time(start_time)
                .add_end_time(end_time)
                .add_since_id(since_id)
                .add_until_id(until_id)
                .add_sort_order(sort_order)
                .build()
            )

            response = cast(
                Response,
                client.get_users_tweets(**params),
            )

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
                self.get_user_tweets(
                    credentials,
                    input_data.user_id,
                    input_data.max_results,
                    input_data.start_time,
                    input_data.end_time,
                    input_data.since_id,
                    input_data.until_id,
                    input_data.sort_order,
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
