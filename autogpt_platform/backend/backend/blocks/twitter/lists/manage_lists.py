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
from backend.blocks.twitter.tweepy_exceptions import handle_tweepy_exception
from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class TwitterDeleteListBlock(Block):
    """
    Deletes a Twitter List owned by the authenticated user
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["list.write", "offline.access"]
        )

        list_id: str = SchemaField(
            description="The ID of the List to be deleted",
            placeholder="Enter list ID",
            required=True,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the deletion was successful")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="843c6892-a62f-11ef-a5c8-b71239a78d3b",
            description="This block deletes a specified Twitter List owned by the authenticated user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterDeleteListBlock.Input,
            output_schema=TwitterDeleteListBlock.Output,
            test_input={"list_id": "1234567890", "credentials": TEST_CREDENTIALS_INPUT},
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"delete_list": lambda *args, **kwargs: True},
        )

    @staticmethod
    def delete_list(credentials: TwitterCredentials, list_id: str):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.delete_list(id=list_id, user_auth=False)
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
            success = self.delete_list(credentials, input_data.list_id)
            yield "success", success

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterUpdateListBlock(Block):
    """
    Updates a Twitter List owned by the authenticated user
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["list.write", "offline.access"]
        )

        list_id: str = SchemaField(
            description="The ID of the List to be updated",
            placeholder="Enter list ID",
            advanced=False,
        )

        name: str | None = SchemaField(
            description="New name for the List",
            placeholder="Enter list name",
            default="",
            advanced=False,
        )

        description: str | None = SchemaField(
            description="New description for the List",
            placeholder="Enter list description",
            default="",
            advanced=False,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether the update was successful")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="7d12630a-a62f-11ef-90c9-8f5a996612c3",
            description="This block updates a specified Twitter List owned by the authenticated user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterUpdateListBlock.Input,
            output_schema=TwitterUpdateListBlock.Output,
            test_input={
                "list_id": "1234567890",
                "name": "Updated List Name",
                "description": "Updated List Description",
                "private": True,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[("success", True)],
            test_mock={"update_list": lambda *args, **kwargs: True},
        )

    @staticmethod
    def update_list(
        credentials: TwitterCredentials,
        list_id: str,
        name: str | None,
        description: str | None,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            client.update_list(
                id=list_id,
                name=None if name == "" else name,
                description=None if description == "" else description,
                user_auth=False,
            )
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
            success = self.update_list(
                credentials, input_data.list_id, input_data.name, input_data.description
            )
            yield "success", success

        except Exception as e:
            yield "error", handle_tweepy_exception(e)


class TwitterCreateListBlock(Block):
    """
    Creates a Twitter List owned by the authenticated user
    """

    class Input(BlockSchema):
        credentials: TwitterCredentialsInput = TwitterCredentialsField(
            ["list.write", "offline.access"]
        )

        name: str = SchemaField(
            description="The name of the List to be created",
            placeholder="Enter list name",
            advanced=False,
            default="",
        )

        description: str | None = SchemaField(
            description="Description of the List",
            placeholder="Enter list description",
            advanced=False,
            default="",
        )

        private: bool = SchemaField(
            description="Whether the List should be private",
            advanced=False,
            default=False,
        )

    class Output(BlockSchema):
        url: str = SchemaField(description="URL of the created list")
        list_id: str = SchemaField(description="ID of the created list")
        error: str = SchemaField(description="Error message if the request failed")

    def __init__(self):
        super().__init__(
            id="724148ba-a62f-11ef-89ba-5349b813ef5f",
            description="This block creates a new Twitter List for the authenticated user.",
            categories={BlockCategory.SOCIAL},
            input_schema=TwitterCreateListBlock.Input,
            output_schema=TwitterCreateListBlock.Output,
            test_input={
                "name": "New List Name",
                "description": "New List Description",
                "private": True,
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("list_id", "1234567890"),
                ("url", "https://twitter.com/i/lists/1234567890"),
            ],
            test_mock={"create_list": lambda *args, **kwargs: ("1234567890")},
        )

    @staticmethod
    def create_list(
        credentials: TwitterCredentials,
        name: str,
        description: str | None,
        private: bool,
    ):
        try:
            client = tweepy.Client(
                bearer_token=credentials.access_token.get_secret_value()
            )

            response = cast(
                Response,
                client.create_list(
                    name=None if name == "" else name,
                    description=None if description == "" else description,
                    private=private,
                    user_auth=False,
                ),
            )

            list_id = str(response.data["id"])

            return list_id

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
            list_id = self.create_list(
                credentials, input_data.name, input_data.description, input_data.private
            )
            yield "list_id", list_id
            yield "url", f"https://twitter.com/i/lists/{list_id}"

        except Exception as e:
            yield "error", handle_tweepy_exception(e)
