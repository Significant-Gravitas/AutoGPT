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


class TwitterGetUserBlock(Block):
    """
    Gets information about a single Twitter user specified by ID or username
    """

    class Input(UserExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["users.read", "offline.access"]
        )

        user_id: str = SchemaField(
            description="The ID of the user to lookup",
            placeholder="Enter user ID",
            default="",
            advanced=False,
        )

        username: str = SchemaField(
            description="The Twitter username (handle) of the user",
            placeholder="Enter username",
            default="",
            advanced=False,
        )

    class Output(BlockSchema):
        # Common outputs
        id: str = SchemaField(description="User ID")
        username_: str = SchemaField(description="User username")
        name_: str = SchemaField(description="User name")

        # Complete outputs
        data: dict = SchemaField(description="Complete user data")
        included: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="5446db8e-a631-11ef-812a-cf315d373ee9",
            description="This block retrieves information about a specified Twitter user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetUserBlock.Input,
            output_schema=TwitterGetUserBlock.Output,
            test_input={
                "user_id": "",
                "username": "twitter",
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": [],
                "tweet_fields": [],
                "user_fields": [],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("id", "783214"),
                ("username_", "twitter"),
                ("name_", "Twitter"),
                (
                    "data",
                    {
                        "user": {
                            "id": "783214",
                            "username": "twitter",
                            "name": "Twitter",
                        }
                    },
                ),
            ],
            test_mock={
                "get_user": lambda *args, **kwargs: (
                    {
                        "user": {
                            "id": "783214",
                            "username": "twitter",
                            "name": "Twitter",
                        }
                    },
                    {},
                    "twitter",
                    "783214",
                    "Twitter",
                )
            },
        )

    @staticmethod
    def get_user(
        credentials: TwitterCredentials,
        user_id: str,
        username: str,
        expansions: UserExpansionsFilter,
        tweet_fields: TweetFieldsFilter,
        user_fields: TweetUserFieldsFilter,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            params = {
                "id": None if not user_id else user_id,
                "username": None if not username else username,
                "user_auth": False,
            }

            params = (
                UserExpansionsBuilder(params)
                .add_expansions(expansions)
                .add_tweet_fields(tweet_fields)
                .add_user_fields(user_fields)
                .build()
            )

            response = cast(Response, client.get_user(**params))

            username = ""
            id = ""
            name = ""

            included = IncludesSerializer.serialize(response.includes)
            data = ResponseDataSerializer.serialize_dict(response.data)

            if response.data:
                username = response.data.username
                id = str(response.data.id)
                name = response.data.name

            if username and id:
                return data, included, username, id, name
            else:
                raise tweepy.TweepyException("User not found")

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
            data, included, username, id, name = self.get_user(
                credentials,
                input_data.user_id,
                input_data.username,
                input_data.expansions,
                input_data.tweet_fields,
                input_data.user_fields,
            )
            if id:
                yield "id", id
            if username:
                yield "username_", username
            if name:
                yield "name_", name
            if data:
                yield "data", data
            if included:
                yield "included", included
        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetUsersBlock(Block):
    """
    Gets information about multiple Twitter users specified by IDs or usernames
    """

    class Input(UserExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["users.read", "offline.access"]
        )

        user_ids: list[str] = SchemaField(
            description="List of user IDs to lookup (max 100)",
            placeholder="Enter user IDs",
            default=[],
            advanced=False,
        )

        usernames: list[str] = SchemaField(
            description="List of Twitter usernames/handles to lookup (max 100)",
            placeholder="Enter usernames",
            default=[],
            advanced=False,
        )

    class Output(BlockSchema):
        # Common outputs
        ids: list[str] = SchemaField(description="User IDs")
        usernames_: list[str] = SchemaField(description="User usernames")
        names_: list[str] = SchemaField(description="User names")

        # Complete outputs
        data: list[dict] = SchemaField(description="Complete users data")
        included: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="5abc857c-a631-11ef-8cfc-f7b79354f7a1",
            description="This block retrieves information about multiple Twitter users.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetUsersBlock.Input,
            output_schema=TwitterGetUsersBlock.Output,
            test_input={
                "user_ids": [],
                "usernames": ["twitter", "twitterdev"],
                "credentials": TEST_CREDENTIALS_INPUT,
                "expansions": [],
                "tweet_fields": [],
                "user_fields": [],
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("ids", ["783214", "2244994945"]),
                ("usernames_", ["twitter", "twitterdev"]),
                ("names_", ["Twitter", "Twitter Dev"]),
                (
                    "data",
                    [
                        {"id": "783214", "username": "twitter", "name": "Twitter"},
                        {
                            "id": "2244994945",
                            "username": "twitterdev",
                            "name": "Twitter Dev",
                        },
                    ],
                ),
            ],
            test_mock={
                "get_users": lambda *args, **kwargs: (
                    [
                        {"id": "783214", "username": "twitter", "name": "Twitter"},
                        {
                            "id": "2244994945",
                            "username": "twitterdev",
                            "name": "Twitter Dev",
                        },
                    ],
                    {},
                    ["twitter", "twitterdev"],
                    ["783214", "2244994945"],
                    ["Twitter", "Twitter Dev"],
                )
            },
        )

    @staticmethod
    def get_users(
        credentials: TwitterCredentials,
        user_ids: list[str],
        usernames: list[str],
        expansions: UserExpansionsFilter,
        tweet_fields: TweetFieldsFilter,
        user_fields: TweetUserFieldsFilter,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            params = {
                "ids": None if not user_ids else user_ids,
                "usernames": None if not usernames else usernames,
                "user_auth": False,
            }

            params = (
                UserExpansionsBuilder(params)
                .add_expansions(expansions)
                .add_tweet_fields(tweet_fields)
                .add_user_fields(user_fields)
                .build()
            )

            response = cast(Response, client.get_users(**params))

            usernames = []
            ids = []
            names = []

            included = IncludesSerializer.serialize(response.includes)
            data = ResponseDataSerializer.serialize_list(response.data)

            if response.data:
                for user in response.data:
                    usernames.append(user.username)
                    ids.append(str(user.id))
                    names.append(user.name)

            if usernames and ids:
                return data, included, usernames, ids, names
            else:
                raise tweepy.TweepyException("Users not found")

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
            data, included, usernames, ids, names = self.get_users(
                credentials,
                input_data.user_ids,
                input_data.usernames,
                input_data.expansions,
                input_data.tweet_fields,
                input_data.user_fields,
            )
            if ids:
                yield "ids", ids
            if usernames:
                yield "usernames_", usernames
            if names:
                yield "names_", names
            if data:
                yield "data", data
            if included:
                yield "included", included
        except Exception as e:
            yield "error", handle_tweepy_exception(e)
