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
from backend.blocks.twitter._builders import TweetExpansionsBuilder
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
)
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TwitterGetTweetBlock(Block):
    """
    Returns information about a single Tweet specified by the requested ID
    """

    class Input(TweetExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "offline.access"]
        )

        tweet_id: str = SchemaField(
            description="Unique identifier of the Tweet to request (ex: 1460323737035677698)",
            placeholder="Enter tweet ID",
        )

    class Output(BlockSchema):
        # Common Outputs that user commonly uses
        id: str = SchemaField(description="Tweet ID")
        text: str = SchemaField(description="Tweet text")
        userId: str = SchemaField(description="ID of the tweet author")
        userName: str = SchemaField(description="Username of the tweet author")

        # Complete Outputs for advanced use
        data: dict = SchemaField(description="Tweet data")
        included: dict = SchemaField(
            description="Additional data that you have requested (Optional) via Expansions field"
        )
        meta: dict = SchemaField(description="Metadata about the tweet")

        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="f5155c3a-a630-11ef-9cc1-a309988b4d92",
            description="This block retrieves information about a specific Tweet.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetTweetBlock.Input,
            output_schema=TwitterGetTweetBlock.Output,
            test_input={
                "tweet_id": "1460323737035677698",
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
                ("id", "1460323737035677698"),
                ("text", "Test tweet content"),
                ("userId", "12345"),
                ("userName", "testuser"),
                ("data", {"id": "1460323737035677698", "text": "Test tweet content"}),
                ("included", {"users": [{"id": "12345", "username": "testuser"}]}),
                ("meta", {"result_count": 1}),
            ],
            test_mock={
                "get_tweet": lambda *args, **kwargs: (
                    {"id": "1460323737035677698", "text": "Test tweet content"},
                    {"users": [{"id": "12345", "username": "testuser"}]},
                    {"result_count": 1},
                    "12345",
                    "testuser",
                )
            },
        )

    @staticmethod
    def get_tweet(
        credentials: TwitterCredentials,
        tweet_id: str,
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
            params = {"id": tweet_id, "user_auth": False}

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

            response = cast(Response, client.get_tweet(**params))

            meta = {}
            user_id = ""
            user_name = ""

            if response.meta:
                meta = response.meta

            included = IncludesSerializer.serialize(response.includes)
            data = ResponseDataSerializer.serialize_dict(response.data)

            if included and "users" in included:
                user_id = str(included["users"][0]["id"])
                user_name = included["users"][0]["username"]

            if response.data:
                return data, included, meta, user_id, user_name

            raise Exception("Tweet not found")

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

            tweet_data, included, meta, user_id, user_name = self.get_tweet(
                credentials,
                input_data.tweet_id,
                input_data.expansions,
                input_data.media_fields,
                input_data.place_fields,
                input_data.poll_fields,
                input_data.tweet_fields,
                input_data.user_fields,
            )

            yield "id", str(tweet_data["id"])
            yield "text", tweet_data["text"]
            if user_id:
                yield "userId", user_id
            if user_name:
                yield "userName", user_name
            yield "data", tweet_data
            if included:
                yield "included", included
            if meta:
                yield "meta", meta

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetTweetsBlock(Block):
    """
    Returns information about multiple Tweets specified by the requested IDs
    """

    class Input(TweetExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "offline.access"]
        )

        tweet_ids: list[str] = SchemaField(
            description="List of Tweet IDs to request (up to 100)",
            placeholder="Enter tweet IDs",
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

        # Complete Outputs for advanced use
        data: list[dict] = SchemaField(description="Complete Tweet data")
        included: dict = SchemaField(
            description="Additional data that you have requested (Optional) via Expansions field"
        )
        meta: dict = SchemaField(description="Metadata about the tweets")

        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="e7cc5420-a630-11ef-bfaf-13bdd8096a51",
            description="This block retrieves information about multiple Tweets.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetTweetsBlock.Input,
            output_schema=TwitterGetTweetsBlock.Output,
            test_input={
                "tweet_ids": ["1460323737035677698"],
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
                ("ids", ["1460323737035677698"]),
                ("texts", ["Test tweet content"]),
                ("userIds", ["67890"]),
                ("userNames", ["testuser1"]),
                ("data", [{"id": "1460323737035677698", "text": "Test tweet content"}]),
                ("included", {"users": [{"id": "67890", "username": "testuser1"}]}),
                ("meta", {"result_count": 1}),
            ],
            test_mock={
                "get_tweets": lambda *args, **kwargs: (
                    ["1460323737035677698"],  # ids
                    ["Test tweet content"],  # texts
                    ["67890"],  # user_ids
                    ["testuser1"],  # user_names
                    [
                        {"id": "1460323737035677698", "text": "Test tweet content"}
                    ],  # data
                    {"users": [{"id": "67890", "username": "testuser1"}]},  # included
                    {"result_count": 1},  # meta
                )
            },
        )

    @staticmethod
    def get_tweets(
        credentials: TwitterCredentials,
        tweet_ids: list[str],
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
            params = {"ids": tweet_ids, "user_auth": False}

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

            response = cast(Response, client.get_tweets(**params))

            if not response.data and not response.meta:
                raise Exception("No tweets found")

            tweet_ids = []
            tweet_texts = []
            user_ids = []
            user_names = []
            meta = {}

            included = IncludesSerializer.serialize(response.includes)
            data = ResponseDataSerializer.serialize_list(response.data)

            if response.data:
                tweet_ids = [str(tweet.id) for tweet in response.data]
                tweet_texts = [tweet.text for tweet in response.data]

            if included and "users" in included:
                for user in included["users"]:
                    user_ids.append(str(user["id"]))
                    user_names.append(user["username"])

            if response.meta:
                meta = response.meta

            return tweet_ids, tweet_texts, user_ids, user_names, data, included, meta

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
            ids, texts, user_ids, user_names, data, included, meta = self.get_tweets(
                credentials,
                input_data.tweet_ids,
                input_data.expansions,
                input_data.media_fields,
                input_data.place_fields,
                input_data.poll_fields,
                input_data.tweet_fields,
                input_data.user_fields,
            )
            if ids:
                yield "ids", ids
            if texts:
                yield "texts", texts
            if user_ids:
                yield "userIds", user_ids
            if user_names:
                yield "userNames", user_names
            if data:
                yield "data", data
            if included:
                yield "included", included
            if meta:
                yield "meta", meta

        except Exception as e:
            yield "error", handle_tweepy_exception(e)
