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


class TwitterGetListTweetsBlock(Block):
    """
    Gets tweets from a specified Twitter list
    """

    class Input(TweetExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "offline.access"]
        )

        list_id: str = SchemaField(
            description="The ID of the List whose Tweets you would like to retrieve",
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
            description="Token for paginating through results",
            placeholder="Enter pagination token",
            default="",
            advanced=True,
        )

    class Output(BlockSchema):
        # Common outputs
        tweet_ids: list[str] = SchemaField(description="List of tweet IDs")
        texts: list[str] = SchemaField(description="List of tweet texts")
        next_token: str = SchemaField(description="Token for next page of results")

        # Complete outputs
        data: list[dict] = SchemaField(description="Complete list tweets data")
        included: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        meta: dict = SchemaField(
            description="Response metadata including pagination tokens"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="6657edb0-a62f-11ef-8c10-0326d832467d",
            description="This block retrieves tweets from a specified Twitter list.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetListTweetsBlock.Input,
            output_schema=TwitterGetListTweetsBlock.Output,
            test_input={
                "list_id": "84839422",
                "max_results": 1,
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
                ("tweet_ids", ["1234567890"]),
                ("texts", ["Test tweet"]),
                ("data", [{"id": "1234567890", "text": "Test tweet"}]),
            ],
            test_mock={
                "get_list_tweets": lambda *args, **kwargs: (
                    [{"id": "1234567890", "text": "Test tweet"}],
                    {},
                    {},
                    ["1234567890"],
                    ["Test tweet"],
                    None,
                )
            },
        )

    @staticmethod
    def get_list_tweets(
        credentials: TwitterCredentials,
        list_id: str,
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
                "id": list_id,
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

            response = cast(Response, client.get_list_tweets(**params))

            meta = {}
            included = {}
            tweet_ids = []
            texts = []
            next_token = None

            if response.meta:
                meta = response.meta
                next_token = meta.get("next_token")

            if response.includes:
                included = IncludesSerializer.serialize(response.includes)

            if response.data:
                data = ResponseDataSerializer.serialize_list(response.data)
                tweet_ids = [str(item.id) for item in response.data]
                texts = [item.text for item in response.data]

                return data, included, meta, tweet_ids, texts, next_token

            raise Exception("No tweets found in this list")

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
            list_data, included, meta, tweet_ids, texts, next_token = (
                self.get_list_tweets(
                    credentials,
                    input_data.list_id,
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

            if tweet_ids:
                yield "tweet_ids", tweet_ids
            if texts:
                yield "texts", texts
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
