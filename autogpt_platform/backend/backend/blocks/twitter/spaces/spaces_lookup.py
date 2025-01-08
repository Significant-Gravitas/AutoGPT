from typing import Literal, Union, cast

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
    SpaceExpansionsBuilder,
    TweetExpansionsBuilder,
    UserExpansionsBuilder,
)
from backend.blocks.twitter._serializer import (
    IncludesSerializer,
    ResponseDataSerializer,
)
from backend.blocks.twitter._types import (
    ExpansionFilter,
    SpaceExpansionInputs,
    SpaceExpansionsFilter,
    SpaceFieldsFilter,
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


class SpaceList(BaseModel):
    discriminator: Literal["space_list"]
    space_ids: list[str] = SchemaField(
        description="List of Space IDs to lookup (up to 100)",
        placeholder="Enter Space IDs",
        default=[],
        advanced=False,
    )


class UserList(BaseModel):
    discriminator: Literal["user_list"]
    user_ids: list[str] = SchemaField(
        description="List of user IDs to lookup their Spaces (up to 100)",
        placeholder="Enter user IDs",
        default=[],
        advanced=False,
    )


class TwitterGetSpacesBlock(Block):
    """
    Gets information about multiple Twitter Spaces specified by Space IDs or creator user IDs
    """

    class Input(SpaceExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["spaces.read", "users.read", "offline.access"]
        )

        identifier: Union[SpaceList, UserList] = SchemaField(
            discriminator="discriminator",
            description="Choose whether to lookup spaces by their IDs or by creator user IDs",
            advanced=False,
        )

    class Output(BlockSchema):
        # Common outputs
        ids: list[str] = SchemaField(description="List of space IDs")
        titles: list[str] = SchemaField(description="List of space titles")

        # Complete outputs for advanced use
        data: list[dict] = SchemaField(description="Complete space data")
        includes: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="d75bd7d8-a62f-11ef-b0d8-c7a9496f617f",
            description="This block retrieves information about multiple Twitter Spaces.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetSpacesBlock.Input,
            output_schema=TwitterGetSpacesBlock.Output,
            test_input={
                "identifier": {
                    "discriminator": "space_list",
                    "space_ids": ["1DXxyRYNejbKM"],
                },
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": None,
                "space_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["1DXxyRYNejbKM"]),
                ("titles", ["Test Space"]),
                (
                    "data",
                    [
                        {
                            "id": "1DXxyRYNejbKM",
                            "title": "Test Space",
                            "host_id": "1234567",
                        }
                    ],
                ),
            ],
            test_mock={
                "get_spaces": lambda *args, **kwargs: (
                    [
                        {
                            "id": "1DXxyRYNejbKM",
                            "title": "Test Space",
                            "host_id": "1234567",
                        }
                    ],
                    {},
                    ["1DXxyRYNejbKM"],
                    ["Test Space"],
                )
            },
        )

    @staticmethod
    def get_spaces(
        credentials: TwitterCredentials,
        identifier: Union[SpaceList, UserList],
        expansions: SpaceExpansionsFilter | None,
        space_fields: SpaceFieldsFilter | None,
        user_fields: TweetUserFieldsFilter | None,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            params = {
                "ids": (
                    identifier.space_ids if isinstance(identifier, SpaceList) else None
                ),
                "user_ids": (
                    identifier.user_ids if isinstance(identifier, UserList) else None
                ),
            }

            params = (
                SpaceExpansionsBuilder(params)
                .add_expansions(expansions)
                .add_space_fields(space_fields)
                .add_user_fields(user_fields)
                .build()
            )

            response = cast(Response, client.get_spaces(**params))

            ids = []
            titles = []

            included = IncludesSerializer.serialize(response.includes)

            if response.data:
                data = ResponseDataSerializer.serialize_list(response.data)
                ids = [space["id"] for space in data if "id" in space]
                titles = [space["title"] for space in data if "title" in space]

                return data, included, ids, titles

            raise Exception("No spaces found")

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
            data, included, ids, titles = self.get_spaces(
                credentials,
                input_data.identifier,
                input_data.expansions,
                input_data.space_fields,
                input_data.user_fields,
            )

            if ids:
                yield "ids", ids
            if titles:
                yield "titles", titles

            if data:
                yield "data", data
            if included:
                yield "includes", included

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetSpaceByIdBlock(Block):
    """
    Gets information about a single Twitter Space specified by Space ID
    """

    class Input(SpaceExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["spaces.read", "users.read", "offline.access"]
        )

        space_id: str = SchemaField(
            description="Space ID to lookup",
            placeholder="Enter Space ID",
            required=True,
        )

    class Output(BlockSchema):
        # Common outputs
        id: str = SchemaField(description="Space ID")
        title: str = SchemaField(description="Space title")
        host_ids: list[str] = SchemaField(description="Host ID")

        # Complete outputs for advanced use
        data: dict = SchemaField(description="Complete space data")
        includes: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="c79700de-a62f-11ef-ab20-fb32bf9d5a9d",
            description="This block retrieves information about a single Twitter Space.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetSpaceByIdBlock.Input,
            output_schema=TwitterGetSpaceByIdBlock.Output,
            test_input={
                "space_id": "1DXxyRYNejbKM",
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": None,
                "space_fields": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("id", "1DXxyRYNejbKM"),
                ("title", "Test Space"),
                ("host_ids", ["1234567"]),
                (
                    "data",
                    {
                        "id": "1DXxyRYNejbKM",
                        "title": "Test Space",
                        "host_ids": ["1234567"],
                    },
                ),
            ],
            test_mock={
                "get_space": lambda *args, **kwargs: (
                    {
                        "id": "1DXxyRYNejbKM",
                        "title": "Test Space",
                        "host_ids": ["1234567"],
                    },
                    {},
                )
            },
        )

    @staticmethod
    def get_space(
        credentials: TwitterCredentials,
        space_id: str,
        expansions: SpaceExpansionsFilter | None,
        space_fields: SpaceFieldsFilter | None,
        user_fields: TweetUserFieldsFilter | None,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            params = {
                "id": space_id,
            }

            params = (
                SpaceExpansionsBuilder(params)
                .add_expansions(expansions)
                .add_space_fields(space_fields)
                .add_user_fields(user_fields)
                .build()
            )

            response = cast(Response, client.get_space(**params))

            includes = {}
            if response.includes:
                for key, value in response.includes.items():
                    if isinstance(value, list):
                        includes[key] = [
                            item.data if hasattr(item, "data") else item
                            for item in value
                        ]
                    else:
                        includes[key] = value.data if hasattr(value, "data") else value

            data = {}
            if response.data:
                for key, value in response.data.items():
                    if isinstance(value, list):
                        data[key] = [
                            item.data if hasattr(item, "data") else item
                            for item in value
                        ]
                    else:
                        data[key] = value.data if hasattr(value, "data") else value

                return data, includes

            raise Exception("Space not found")

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
            space_data, includes = self.get_space(
                credentials,
                input_data.space_id,
                input_data.expansions,
                input_data.space_fields,
                input_data.user_fields,
            )

            # Common outputs
            if space_data:
                if "id" in space_data:
                    yield "id", space_data.get("id")

                if "title" in space_data:
                    yield "title", space_data.get("title")

                if "host_ids" in space_data:
                    yield "host_ids", space_data.get("host_ids")

            if space_data:
                yield "data", space_data
            if includes:
                yield "includes", includes

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


# Not tested yet, might have some problem
class TwitterGetSpaceBuyersBlock(Block):
    """
    Gets list of users who purchased a ticket to the requested Space
    """

    class Input(UserExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["spaces.read", "users.read", "offline.access"]
        )

        space_id: str = SchemaField(
            description="Space ID to lookup buyers for",
            placeholder="Enter Space ID",
            required=True,
        )

    class Output(BlockSchema):
        # Common outputs
        buyer_ids: list[str] = SchemaField(description="List of buyer IDs")
        usernames: list[str] = SchemaField(description="List of buyer usernames")

        # Complete outputs for advanced use
        data: list[dict] = SchemaField(description="Complete space buyers data")
        includes: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="c1c121a8-a62f-11ef-8b0e-d7b85f96a46f",
            description="This block retrieves a list of users who purchased tickets to a Twitter Space.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetSpaceBuyersBlock.Input,
            output_schema=TwitterGetSpaceBuyersBlock.Output,
            test_input={
                "space_id": "1DXxyRYNejbKM",
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": None,
                "user_fields": None,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("buyer_ids", ["2244994945"]),
                ("usernames", ["testuser"]),
                (
                    "data",
                    [{"id": "2244994945", "username": "testuser", "name": "Test User"}],
                ),
            ],
            test_mock={
                "get_space_buyers": lambda *args, **kwargs: (
                    [{"id": "2244994945", "username": "testuser", "name": "Test User"}],
                    {},
                    ["2244994945"],
                    ["testuser"],
                )
            },
        )

    @staticmethod
    def get_space_buyers(
        credentials: TwitterCredentials,
        space_id: str,
        expansions: UserExpansionsFilter | None,
        user_fields: TweetUserFieldsFilter | None,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            params = {
                "id": space_id,
            }

            params = (
                UserExpansionsBuilder(params)
                .add_expansions(expansions)
                .add_user_fields(user_fields)
                .build()
            )

            response = cast(Response, client.get_space_buyers(**params))

            included = IncludesSerializer.serialize(response.includes)

            if response.data:
                data = ResponseDataSerializer.serialize_list(response.data)
                buyer_ids = [buyer["id"] for buyer in data]
                usernames = [buyer["username"] for buyer in data]

                return data, included, buyer_ids, usernames

            raise Exception("No buyers found for this Space")

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
            buyers_data, included, buyer_ids, usernames = self.get_space_buyers(
                credentials,
                input_data.space_id,
                input_data.expansions,
                input_data.user_fields,
            )

            if buyer_ids:
                yield "buyer_ids", buyer_ids
            if usernames:
                yield "usernames", usernames

            if buyers_data:
                yield "data", buyers_data
            if included:
                yield "includes", included

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetSpaceTweetsBlock(Block):
    """
    Gets list of Tweets shared in the requested Space
    """

    class Input(TweetExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["spaces.read", "users.read", "offline.access"]
        )

        space_id: str = SchemaField(
            description="Space ID to lookup tweets for",
            placeholder="Enter Space ID",
            required=True,
        )

    class Output(BlockSchema):
        # Common outputs
        tweet_ids: list[str] = SchemaField(description="List of tweet IDs")
        texts: list[str] = SchemaField(description="List of tweet texts")

        # Complete outputs for advanced use
        data: list[dict] = SchemaField(description="Complete space tweets data")
        includes: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        meta: dict = SchemaField(description="Response metadata")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="b69731e6-a62f-11ef-b2d4-1bf14dd6aee4",
            description="This block retrieves tweets shared in a Twitter Space.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetSpaceTweetsBlock.Input,
            output_schema=TwitterGetSpaceTweetsBlock.Output,
            test_input={
                "space_id": "1DXxyRYNejbKM",
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
                "get_space_tweets": lambda *args, **kwargs: (
                    [{"id": "1234567890", "text": "Test tweet"}],  # data
                    {},
                    ["1234567890"],
                    ["Test tweet"],
                    {},
                )
            },
        )

    @staticmethod
    def get_space_tweets(
        credentials: TwitterCredentials,
        space_id: str,
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
                "id": space_id,
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

            response = cast(Response, client.get_space_tweets(**params))

            included = IncludesSerializer.serialize(response.includes)

            if response.data:
                data = ResponseDataSerializer.serialize_list(response.data)
                tweet_ids = [str(tweet["id"]) for tweet in data]
                texts = [tweet["text"] for tweet in data]

                meta = response.meta or {}

                return data, included, tweet_ids, texts, meta

            raise Exception("No tweets found for this Space")

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
            tweets_data, included, tweet_ids, texts, meta = self.get_space_tweets(
                credentials,
                input_data.space_id,
                input_data.expansions,
                input_data.media_fields,
                input_data.place_fields,
                input_data.poll_fields,
                input_data.tweet_fields,
                input_data.user_fields,
            )

            if tweet_ids:
                yield "tweet_ids", tweet_ids
            if texts:
                yield "texts", texts

            if tweets_data:
                yield "data", tweets_data
            if included:
                yield "includes", included
            if meta:
                yield "meta", meta

        except Exception as e:
            yield "error", handle_tweepy_exception(e)
