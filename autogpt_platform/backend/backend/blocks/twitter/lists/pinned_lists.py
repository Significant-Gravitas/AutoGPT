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
from backend.blocks.twitter._builders import ListExpansionsBuilder
from backend.blocks.twitter._serializer import (
    IncludesSerializer,
    ResponseDataSerializer,
)
from backend.blocks.twitter._types import (
    ListExpansionInputs,
    ListExpansionsFilter,
    ListFieldsFilter,
    TweetUserFieldsFilter,
)
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TwitterUnpinListBlock(Block):
    """
    Enables the authenticated user to unpin a List.
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["list.write", "users.read", "tweet.read", "offline.access"]
        )

        list_id: str = SchemaField(
            description="The ID of the List to unpin",
            placeholder="Enter list ID",
            required=True,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the unpin was successful")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="a099c034-a62f-11ef-9622-47d0ceb73555",
            description="This block allows the authenticated user to unpin a specified List.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterUnpinListBlock.Input,
            output_schema=TwitterUnpinListBlock.Output,
            test_input={"list_id": "123456789", "credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"unpin_list": lambda *args, **kwargs: True},
        )

    @staticmethod
    def unpin_list(credentials: TwitterCredentials, list_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.unpin_list(list_id=list_id, user_auth=False)

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
            success = self.unpin_list(credentials, input_data.list_id)
            yield "success", success

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterPinListBlock(Block):
    """
    Enables the authenticated user to pin a List.
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["list.write", "users.read", "tweet.read", "offline.access"]
        )

        list_id: str = SchemaField(
            description="The ID of the List to pin",
            placeholder="Enter list ID",
            required=True,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the pin was successful")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="8ec16e48-a62f-11ef-9f35-f3d6de43a802",
            description="This block allows the authenticated user to pin a specified List.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterPinListBlock.Input,
            output_schema=TwitterPinListBlock.Output,
            test_input={"list_id": "123456789", "credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"pin_list": lambda *args, **kwargs: True},
        )

    @staticmethod
    def pin_list(credentials: TwitterCredentials, list_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.pin_list(list_id=list_id, user_auth=False)

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
            success = self.pin_list(credentials, input_data.list_id)
            yield "success", success

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterGetPinnedListsBlock(Block):
    """
    Returns the Lists pinned by the authenticated user.
    """

    class Input(ListExpansionInputs):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["lists.read", "users.read", "offline.access"]
        )

    class Output(BlockSchema):
        list_ids: list[str] = SchemaField(description="List IDs of the pinned lists")
        list_names: list[str] = SchemaField(
            description="List names of the pinned lists"
        )

        data: list[dict] = SchemaField(
            description="Response data containing pinned lists"
        )
        included: dict = SchemaField(
            description="Additional data requested via expansions"
        )
        meta: dict = SchemaField(description="Metadata about the response")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="97e03aae-a62f-11ef-bc53-5b89cb02888f",
            description="This block returns the Lists pinned by the authenticated user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterGetPinnedListsBlock.Input,
            output_schema=TwitterGetPinnedListsBlock.Output,
            test_input={
                "expansions": None,
                "list_fields": None,
                "user_fields": None,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("list_ids", ["84839422"]),
                ("list_names", ["Twitter List"]),
                ("data", [{"id": "84839422", "name": "Twitter List"}]),
            ],
            test_mock={
                "get_pinned_lists": lambda *args, **kwargs: (
                    [{"id": "84839422", "name": "Twitter List"}],
                    {},
                    {},
                    ["84839422"],
                    ["Twitter List"],
                )
            },
        )

    @staticmethod
    def get_pinned_lists(
        credentials: TwitterCredentials,
        expansions: ListExpansionsFilter | None,
        user_fields: TweetUserFieldsFilter | None,
        list_fields: ListFieldsFilter | None,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            params = {"user_auth": False}

            params = (
                ListExpansionsBuilder(params)
                .add_expansions(expansions)
                .add_user_fields(user_fields)
                .add_list_fields(list_fields)
                .build()
            )

            response = cast(Response, client.get_pinned_lists(**params))

            meta = {}
            included = {}
            list_ids = []
            list_names = []

            if response.meta:
                meta = response.meta

            if response.includes:
                included = IncludesSerializer.serialize(response.includes)

            if response.data:
                data = ResponseDataSerializer.serialize_list(response.data)
                list_ids = [str(item.id) for item in response.data]
                list_names = [item.name for item in response.data]
                return data, included, meta, list_ids, list_names

            raise Exception("Lists not found")

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
            list_data, included, meta, list_ids, list_names = self.get_pinned_lists(
                credentials,
                input_data.expansions,
                input_data.user_fields,
                input_data.list_fields,
            )

            if list_ids:
                yield "list_ids", list_ids
            if list_names:
                yield "list_names", list_names
            if list_data:
                yield "data", list_data
            if included:
                yield "included", included
            if meta:
                yield "meta", meta

        except Exception as e:
            yield "error", handle_tweepy_exception(e)
