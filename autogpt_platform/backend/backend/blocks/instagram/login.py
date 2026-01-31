"""
Instagram Login Block for AutoGPT Platform.
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


class InstagramLoginBlock(Block):
    """
    Logs into Instagram and returns a session.

    This block authenticates with Instagram using username and password.
    The credentials should be stored as 'username:password' in the API key field.
    """

    class Input(BlockSchemaInput):
        credentials: InstagramCredentialsInput = InstagramCredentialsField()

    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether login was successful")
        user_id: str = SchemaField(
            description="Instagram user ID of the logged-in account"
        )
        username: str = SchemaField(
            description="Instagram username of the logged-in account"
        )
        error: str = SchemaField(
            description="Error message if login failed", default=""
        )

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description="Login to Instagram using username and password credentials",
            categories={BlockCategory.SOCIAL},
            input_schema=InstagramLoginBlock.Input,
            output_schema=InstagramLoginBlock.Output,
            test_input={
                "credentials": TEST_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_CREDENTIALS,
            test_output=[
                ("success", True),
                ("user_id", "123456789"),
                ("username", "test_username"),
            ],
            test_mock={
                "login": lambda *args, **kwargs: (
                    True,
                    "123456789",
                    "test_username",
                    None,
                )
            },
        )

    @staticmethod
    def login(credentials: InstagramCredentials):
        """
        Login to Instagram.

        Args:
            credentials: Instagram credentials containing username:password

        Returns:
            Tuple of (success, user_id, username, error)
        """
        try:
            # Extract username and password from API key
            api_key = credentials.api_key.get_secret_value()
            if ":" not in api_key:
                return (
                    False,
                    None,
                    None,
                    "Invalid credentials format. Use 'username:password'",
                )

            username, password = api_key.split(":", 1)

            # Create Instagram client
            client = Client()
            client.login(username, password)

            # Get user info
            user_id = str(client.user_id)
            username = client.username

            return True, user_id, username, None

        except Exception as e:
            return False, None, None, f"Login failed: {str(e)}"

    async def run(
        self,
        input_data: Input,
        *,
        credentials: InstagramCredentials,
        **kwargs,
    ) -> BlockOutput:
        """
        Execute the Instagram login.

        Args:
            input_data: Input data containing credentials reference
            credentials: Actual Instagram credentials
            **kwargs: Additional arguments

        Yields:
            Block outputs: success, user_id, username, or error
        """
        success, user_id, username, error = self.login(credentials)

        if success:
            yield "success", success
            if user_id:
                yield "user_id", user_id
            if username:
                yield "username", username
        else:
            yield "success", False
            yield "error", error or "Unknown error occurred"
