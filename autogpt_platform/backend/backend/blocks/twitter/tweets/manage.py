from datetime import datetime
from typing import List, Literal, Optional, Union, cast

import tweepy
from pydantic import BaseModel
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
    TweetPostBuilder,
    TweetSearchBuilder,
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
    TweetReplySettingsFilter,
    TweetTimeWindowInputs,
    TweetUserFieldsFilter,
)
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class Media(BaseModel):
    discriminator: Literal["media"]
    media_ids: Optional[List[str]] = None
    media_tagged_user_ids: Optional[List[str]] = None


class DeepLink(BaseModel):
    discriminator: Literal["deep_link"]
    direct_message_deep_link: Optional[str] = None


class Poll(BaseModel):
    discriminator: Literal["poll"]
    poll_options: Optional[List[str]] = None
    poll_duration_minutes: Optional[int] = None


class Place(BaseModel):
    discriminator: Literal["place"]
    place_id: Optional[str] = None


class Quote(BaseModel):
    discriminator: Literal["quote"]
    quote_tweet_id: Optional[str] = None


class TwitterPostTweetBlock(Block):
    """
    Create a tweet on Twitter with the option to include one additional element such as a media, quote, or deep link.
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "tweet.write", "users.read", "offline.access"]
        )

        tweet_text: str | None = SchemaField(
            description="Text of the tweet to post",
            placeholder="Enter your tweet",
            default=None,
            advanced=False,
        )

        for_super_followers_only: bool = SchemaField(
            description="Tweet exclusively for Super Followers",
            placeholder="Enter for super followers only",
            advanced=True,
            default=False,
        )

        attachment: Union[Media, DeepLink, Poll, Place, Quote] | None = SchemaField(
            discriminator="discriminator",
            description="Additional tweet data (media, deep link, poll, place or quote)",
            advanced=False,
            default=Media(discriminator="media"),
        )

        exclude_reply_user_ids: Optional[List[str]] = SchemaField(
            description="User IDs to exclude from reply Tweet thread. [ex - 6253282]",
            placeholder="Enter user IDs to exclude",
            advanced=True,
            default=None,
        )

        in_reply_to_tweet_id: Optional[str] = SchemaField(
            description="Tweet ID being replied to. Please note that in_reply_to_tweet_id needs to be in the request if exclude_reply_user_ids is present",
            default=None,
            placeholder="Enter in reply to tweet ID",
            advanced=True,
        )

        reply_settings: TweetReplySettingsFilter = SchemaField(
            description="Who can reply to the Tweet (mentionedUsers or following)",
            placeholder="Enter reply settings",
            advanced=True,
            default=TweetReplySettingsFilter(All_Users=True),
        )

    class Output(BlockSchema):
        tweet_id: str = SchemaField(description="ID of the created tweet")
        tweet_url: str = SchemaField(description="URL to the tweet")
        error: str = SchemaField(
            description="Error message if the tweet posting failed"
        )

    def __init__(self):
        super().__init__(
            id="7bb0048a-a630-11ef-aeb8-abc0dadb9b12",
            description="This block posts a tweet on Twitter.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterPostTweetBlock.Input,
            output_schema=TwitterPostTweetBlock.Output,
            test_input={
                "tweet_text": "This is a test tweet.",
                "credentials": TEST_CREDENTIALS_INPUT,
                "attachment": {
                    "discriminator": "deep_link",
                    "direct_message_deep_link": "https://twitter.com/messages/compose",
                },
                "for_super_followers_only": False,
                "exclude_reply_user_ids": [],
                "in_reply_to_tweet_id": "",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("tweet_id", "1234567890"),
                ("tweet_url", "https://twitter.com/user/status/1234567890"),
            ],
            test_mock={
                "post_tweet": lambda *args, **kwargs: (
                    "1234567890",
                    "https://twitter.com/user/status/1234567890",
                )
            },
        )

    def post_tweet(
        self,
        credentials: TwitterCredentials,
        input_txt: str | None,
        attachment: Union[Media, DeepLink, Poll, Place, Quote] | None,
        for_super_followers_only: bool,
        exclude_reply_user_ids: Optional[List[str]],
        in_reply_to_tweet_id: Optional[str],
        reply_settings: TweetReplySettingsFilter,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            params = (
                TweetPostBuilder()
                .add_text(input_txt)
                .add_super_followers(for_super_followers_only)
                .add_reply_settings(
                    exclude_reply_user_ids or [],
                    in_reply_to_tweet_id or "",
                    reply_settings,
                )
            )

            if isinstance(attachment, Media):
                params.add_media(
                    attachment.media_ids or [], attachment.media_tagged_user_ids or []
                )
            elif isinstance(attachment, DeepLink):
                params.add_deep_link(attachment.direct_message_deep_link or "")
            elif isinstance(attachment, Poll):
                params.add_poll_options(attachment.poll_options or [])
                params.add_poll_duration(attachment.poll_duration_minutes or 0)
            elif isinstance(attachment, Place):
                params.add_place(attachment.place_id or "")
            elif isinstance(attachment, Quote):
                params.add_quote(attachment.quote_tweet_id or "")

            tweet = cast(Response, client.create_tweet(**params.build()))

            if not tweet.data:
                raise Exception("Failed to create tweet")

            tweet_id = tweet.data["id"]
            tweet_url = f"https://twitter.com/user/status/{tweet_id}"
            return str(tweet_id), tweet_url

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
            tweet_id, tweet_url = self.post_tweet(
                credentials,
                input_data.tweet_text,
                input_data.attachment,
                input_data.for_super_followers_only,
                input_data.exclude_reply_user_ids,
                input_data.in_reply_to_tweet_id,
                input_data.reply_settings,
            )
            yield "tweet_id", tweet_id
            yield "tweet_url", tweet_url

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterDeleteTweetBlock(Block):
    """
    Deletes a tweet on Twitter using twitter Id
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "tweet.write", "users.read", "offline.access"]
        )

        tweet_id: str = SchemaField(
            description="ID of the tweet to delete",
            placeholder="Enter tweet ID",
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether the tweet was successfully deleted"
        )
        error: str = SchemaField(
            description="Error message if the tweet deletion failed"
        )

    def __init__(self):
        super().__init__(
            id="761babf0-a630-11ef-a03d-abceb082f58f",
            description="This block deletes a tweet on Twitter.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterDeleteTweetBlock.Input,
            output_schema=TwitterDeleteTweetBlock.Output,
            test_input={
                "tweet_id": "1234567890",
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"delete_tweet": lambda *args, **kwargs: True},
        )

    @staticmethod
    def delete_tweet(credentials: TwitterCredentials, tweet_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )
            client.delete_tweet(id=tweet_id, user_auth=False)
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
            success = self.delete_tweet(
                credentials,
                input_data.tweet_id,
            )
            yield "success", success

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterSearchRecentTweetsBlock(Block):
    """
    Searches all public Tweets in Twitter history
    """

    class Input(TweetExpansionInputs, TweetTimeWindowInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["tweet.read", "users.read", "offline.access"]
        )

        query: str = SchemaField(
            description="Search query (up to 1024 characters)",
            placeholder="Enter search query",
        )

        max_results: int = SchemaField(
            description="Maximum number of results per page (10-500)",
            placeholder="Enter max results",
            default=10,
            advanced=True,
        )

        pagination: str | None = SchemaField(
            description="Token for pagination",
            default="",
            placeholder="Enter pagination token",
            advanced=True,
        )

    class Output(BlockSchema):
        # Common Outputs that user commonly uses
        tweet_ids: list[str] = SchemaField(description="All Tweet IDs")
        tweet_texts: list[str] = SchemaField(description="All Tweet texts")
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
            id="53e5cf8e-a630-11ef-ba85-df6d666fa5d5",
            description="This block searches all public Tweets in Twitter history.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterSearchRecentTweetsBlock.Input,
            output_schema=TwitterSearchRecentTweetsBlock.Output,
            test_input={
                "query": "from:twitterapi #twitterapi",
                "credentials": TEST_CREDENTIALS_INPUT,
                "max_results": 2,
                "start_time": "2024-12-14T18:30:00.000Z",
                "end_time": "2024-12-17T18:30:00.000Z",
                "since_id": None,
                "until_id": None,
                "sort_order": None,
                "pagination": None,
                "expansions": None,
                "media_fields": None,
                "place_fields": None,
                "poll_fields": None,
                "tweet_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("tweet_ids", ["1373001119480344583", "1372627771717869568"]),
                (
                    "tweet_texts",
                    [
                        "Looking to get started with the Twitter API but new to APIs in general?",
                        "Thanks to everyone who joined and made today a great session!",
                    ],
                ),
                (
                    "data",
                    [
                        {
                            "id": "1373001119480344583",
                            "text": "Looking to get started with the Twitter API but new to APIs in general?",
                        },
                        {
                            "id": "1372627771717869568",
                            "text": "Thanks to everyone who joined and made today a great session!",
                        },
                    ],
                ),
            ],
            test_mock={
                "search_tweets": lambda *args, **kwargs: (
                    ["1373001119480344583", "1372627771717869568"],
                    [
                        "Looking to get started with the Twitter API but new to APIs in general?",
                        "Thanks to everyone who joined and made today a great session!",
                    ],
                    [
                        {
                            "id": "1373001119480344583",
                            "text": "Looking to get started with the Twitter API but new to APIs in general?",
                        },
                        {
                            "id": "1372627771717869568",
                            "text": "Thanks to everyone who joined and made today a great session!",
                        },
                    ],
                    {},
                    {},
                    None,
                )
            },
        )

    @staticmethod
    def search_tweets(
        credentials: TwitterCredentials,
        query: str,
        max_results: int,
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

            # Building common params
            params = (
                TweetSearchBuilder()
                .add_query(query)
                .add_pagination(max_results, pagination)
                .build()
            )

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

            response = cast(Response, client.search_recent_tweets(**params))

            if not response.data and not response.meta:
                raise Exception("No tweets found")

            meta = {}
            tweet_ids = []
            tweet_texts = []
            next_token = None

            if response.meta:
                meta = response.meta
                next_token = meta.get("next_token")

            included = IncludesSerializer.serialize(response.includes)
            data = ResponseDataSerializer.serialize_list(response.data)

            if response.data:
                tweet_ids = [str(tweet.id) for tweet in response.data]
                tweet_texts = [tweet.text for tweet in response.data]

                return tweet_ids, tweet_texts, data, included, meta, next_token

            raise Exception("No tweets found")

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
            ids, texts, data, included, meta, next_token = self.search_tweets(
                credentials,
                input_data.query,
                input_data.max_results,
                input_data.start_time,
                input_data.end_time,
                input_data.since_id,
                input_data.until_id,
                input_data.sort_order,
                input_data.pagination,
                input_data.expansions,
                input_data.media_fields,
                input_data.place_fields,
                input_data.poll_fields,
                input_data.tweet_fields,
                input_data.user_fields,
            )
            if ids:
                yield "tweet_ids", ids
            if texts:
                yield "tweet_texts", texts
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
