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


class TwitterBookmarkTweetBlock(Block):
    """
    Bookmark a tweet on Twitter
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "bookmark.write", "users.read", "offline.access"]
        )

        tweet_id: str = SchemaField(
            description="ID of the tweet to bookmark",
            placeholder="Enter tweet ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the bookmark was successful")
        error: str = SchemaField(description="Error message if the bookmark failed")

    def __init__(self):
        super().__init__(
            id="f33d67be-a62f-11ef-a797-ff83ec29ee8e",
            description="This block bookmarks a tweet on Twitter.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterBookmarkTweetBlock.Input,
            output_schema=TwitterBookmarkTweetBlock.Output,
            test_input={
                "tweet_id": "1234567890",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"bookmark_tweet": lambda *args, **kwargs: True},
        )

    @staticmethod
    def bookmark_tweet(
        credentials: TwitterCredentials,
        tweet_id: str,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.bookmark(tweet_id)

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
            success = self.bookmark_tweet(credentials, input_data.tweet_id)
            yield "success", success
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetBookmarkedTweetsBlock(Block):
    """
    Get All your bookmarked tweets from Twitter
    """

    class Input(TweetExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "bookmark.read", "users.read", "offline.access"]
        )

        max_results: int | None = SchemaField(
            description="Maximum number of results to return (1-100)",
            placeholder="Enter max results",
            default=10,
            advanced=True,
        )

        pagination_token: str | None = SchemaField(
            description="Token for pagination",
            placeholder="Enter pagination token",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        # Common Outputs that user commonly uses
        id: list[str] = SchemaField(description="All Tweet IDs")
        text: list[str] = SchemaField(description="All Tweet texts")
        userId: list[str] = SchemaField(description="IDs of the tweet authors")
        userName: list[str] = SchemaField(description="Usernames of the tweet authors")

        # Complete Outputs for advanced use
        data: list[dict] = SchemaField(description="Complete Tweet data")
        included: dict = SchemaField(
            description="Additional data that you have requested (Optional) via Expansions field"
        )
        meta: dict = SchemaField(
            description="Provides metadata such as pagination info (next_token) or result counts"
        )
        next_token: str = SchemaField(description="Next token for pagination")

        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="ed26783e-a62f-11ef-9a21-c77c57dd8a1f",
            description="This block retrieves bookmarked tweets from Twitter.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetBookmarkedTweetsBlock.Input,
            output_schema=TwitterGetBookmarkedTweetsBlock.Output,
            test_input={
                "max_results": 2,
                "pagination_token": None,
                "expansions": None,
                "media_fields": None,
                "place_fields": None,
                "poll_fields": None,
                "tweet_fields": None,
                "user_fields": None,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("id", ["1234567890"]),
                ("text", ["Test tweet"]),
                ("userId", ["12345"]),
                ("userName", ["testuser"]),
                ("data", [{"id": "1234567890", "text": "Test tweet"}]),
            ],
            test_mock={
                "get_bookmarked_tweets": lambda *args, **kwargs: (
                    ["1234567890"],
                    ["Test tweet"],
                    ["12345"],
                    ["testuser"],
                    [{"id": "1234567890", "text": "Test tweet"}],
                    {},
                    {},
                    None,
                )
            },
        )

    @staticmethod
    def get_bookmarked_tweets(
        credentials: TwitterCredentials,
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
                "max_results": max_results,
                "pagination_token": (
                    None if pagination_token == "" else pagination_token
                ),
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

            response = cast(
                Response,
                client.get_bookmarks(**params),
            )

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

                if "users" in included:
                    for user in included["users"]:
                        user_ids.append(str(user["id"]))
                        user_names.append(user["username"])

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

            raise Exception("No bookmarked tweets found")

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
                self.get_bookmarked_tweets(
                    credentials,
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
                yield "id", ids
            if texts:
                yield "text", texts
            if user_ids:
                yield "userId", user_ids
            if user_names:
                yield "userName", user_names
            if data:
                yield "data", data
            if included:
                yield "included", included
            if meta:
                yield "meta", meta
            if next_token:
                yield "next_token", next_token
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterRemoveBookmarkTweetBlock(Block):
    """
    Remove a bookmark for a tweet on Twitter
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "bookmark.write", "users.read", "offline.access"]
        )

        tweet_id: str = SchemaField(
            description="ID of the tweet to remove bookmark from",
            placeholder="Enter tweet ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the bookmark was successfully removed"
        )
        error: str = SchemaField(
            description="Error message if the bookmark removal failed"
        )

    def __init__(self):
        super().__init__(
            id="e4100684-a62f-11ef-9be9-770cb41a2616",
            description="This block removes a bookmark from a tweet on Twitter.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterRemoveBookmarkTweetBlock.Input,
            output_schema=TwitterRemoveBookmarkTweetBlock.Output,
            test_input={
                "tweet_id": "1234567890",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
            ],
            test_mock={"remove_bookmark_tweet": lambda *args, **kwargs: True},
        )

    @staticmethod
    def remove_bookmark_tweet(
        credentials: TwitterCredentials,
        tweet_id: str,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.remove_bookmark(tweet_id)

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
            success = self.remove_bookmark_tweet(credentials, input_data.tweet_id)
            yield "success", success
        except Exception as e:
            yield "error", handle_tweepy_exception(e)
