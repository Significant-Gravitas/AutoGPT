"""
Instagram Follow Blocks for AutoGPT Platform.
"""

from instagrapi import Client

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

from .auth import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    InstagramCredentials,
    InstagramCredentialsField,
    InstagramCredentialsInput,
)


class InstagramFollowUserBlock(Block):
    """
    Follows an Instagram user.

    This block follows a user by their username or user ID.
    """

    class Input(BlockSchemaInput):
        credentials: InstagramCredentialsInput = InstagramCredentialsField()

        username: str = SchemaField(
            description="Instagram username to follow (e.g., 'autogpt' or user ID)",
            placeholder="username or user_id",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the follow was successful")
        user_id: str = SchemaField(description="User ID of the followed account")
        error: str = SchemaField(
            description="Error message if following failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="f6a7b8c9-d0e1-2345-f1a2-456789012345",
            description="Follow an Instagram user by username or user ID",
            categories={BlockCategory.SOCIAL},
            input_schema=InstagramFollowUserBlock.Input,
            output_schema=InstagramFollowUserBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "username": "test_user",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
                ("user_id", "987654321"),
            ],
            test_mock={
                "follow_user": lambda *args, **kwargs: (True, "987654321", None)
            },
        )

    @staticmethod
    def follow_user(credentials: InstagramCredentials, username: str):
        """Follow an Instagram user."""
        try:
            api_key = credentials.api_key.get_secret_value()
            if ":" not in api_key:
                return False, None, "Invalid credentials format"

            user, pwd = api_key.split(":", 1)

            client = Client()
            client.login(user, pwd)

            # Get user ID from username if needed
            if not username.isdigit():
                user_id = client.user_id_from_username(username)
            else:
                user_id = int(username)

            # Follow the user
            client.user_follow(user_id)

            return True, str(user_id), None

        except Exception as e:
            return False, None, f"Failed to follow user: {str(e)}"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: InstagramCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the follow action."""
        success, user_id, error = self.follow_user(credentials, input_data.username)

        yield "success", success
        if user_id:
            yield "user_id", user_id
        if error:
            yield "error", error


class InstagramUnfollowUserBlock(Block):
    """
    Unfollows an Instagram user.

    This block unfollows a user by their username or user ID.
    """

    class Input(BlockSchemaInput):
        credentials: InstagramCredentialsInput = InstagramCredentialsField()

        username: str = SchemaField(
            description="Instagram username to unfollow (e.g., 'autogpt' or user ID)",
            placeholder="username or user_id",
        )

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the unfollow was successful")
        user_id: str = SchemaField(description="User ID of the unfollowed account")
        error: str = SchemaField(
            description="Error message if unfollowing failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="a7b8c9d0-e1f2-3456-a2b3-567890123456",
            description="Unfollow an Instagram user by username or user ID",
            categories={BlockCategory.SOCIAL},
            input_schema=InstagramUnfollowUserBlock.Input,
            output_schema=InstagramUnfollowUserBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
                "username": "test_user",
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
                ("user_id", "987654321"),
            ],
            test_mock={
                "unfollow_user": lambda *args, **kwargs: (True, "987654321", None)
            },
        )

    @staticmethod
    def unfollow_user(credentials: InstagramCredentials, username: str):
        """Unfollow an Instagram user."""
        try:
            api_key = credentials.api_key.get_secret_value()
            if ":" not in api_key:
                return False, None, "Invalid credentials format"

            user, pwd = api_key.split(":", 1)

            client = Client()
            client.login(user, pwd)

            # Get user ID from username if needed
            if not username.isdigit():
                user_id = client.user_id_from_username(username)
            else:
                user_id = int(username)

            # Unfollow the user
            client.user_unfollow(user_id)

            return True, str(user_id), None

        except Exception as e:
            return False, None, f"Failed to unfollow user: {str(e)}"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: InstagramCredentials,
        **kwargs,
    ) -> BlockOutput:
        """Execute the unfollow action."""
        success, user_id, error = self.unfollow_user(credentials, input_data.username)

        yield "success", success
        if user_id:
            yield "user_id", user_id
        if error:
            yield "error", error
