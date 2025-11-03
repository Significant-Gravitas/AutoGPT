"""
Discord OAuth-based blocks.
"""

from backend.data.block import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import OAuth2Credentials, SchemaField

from ._api import DiscordOAuthUser, get_current_user
from ._auth import (
    DISCORD_OAUTH_IS_CONFIGURED,
    TEST_OAUTH_CREDENTIALS,
    TEST_OAUTH_CREDENTIALS_INPUT,
    DiscordOAuthCredentialsField,
    DiscordOAuthCredentialsInput,
)


class DiscordGetCurrentUserBlock(Block):
    """
    Gets information about the currently authenticated Discord user using OAuth2.
    This block requires Discord OAuth2 credentials (not bot tokens).
    """

    class Input(BlockSchemaInput):
        credentials: DiscordOAuthCredentialsInput = DiscordOAuthCredentialsField(
            ["identify"]
        )

    class Output(BlockSchemaOutput):
        user_id: str = SchemaField(description="The authenticated user's Discord ID")
        username: str = SchemaField(description="The user's username")
        avatar_url: str = SchemaField(description="URL to the user's avatar image")
        banner_url: str = SchemaField(
            description="URL to the user's banner image (if set)", default=""
        )
        accent_color: int = SchemaField(
            description="The user's accent color as an integer", default=0
        )

    def __init__(self):
        super().__init__(
            id="8c7e39b8-4e9d-4f3a-b4e1-2a8c9d5f6e3b",
            input_schema=DiscordGetCurrentUserBlock.Input,
            output_schema=DiscordGetCurrentUserBlock.Output,
            description="Gets information about the currently authenticated Discord user using OAuth2 credentials.",
            categories={BlockCategory.SOCIAL},
            disabled=not DISCORD_OAUTH_IS_CONFIGURED,
            test_input={
                "credentials": TEST_OAUTH_CREDENTIALS_INPUT,
            },
            test_credentials=TEST_OAUTH_CREDENTIALS,
            test_output=[
                ("user_id", "123456789012345678"),
                ("username", "testuser"),
                (
                    "avatar_url",
                    "https://cdn.discordapp.com/avatars/123456789012345678/avatar.png",
                ),
                ("banner_url", ""),
                ("accent_color", 0),
            ],
            test_mock={
                "get_user": lambda _: DiscordOAuthUser(
                    user_id="123456789012345678",
                    username="testuser",
                    avatar_url="https://cdn.discordapp.com/avatars/123456789012345678/avatar.png",
                    banner=None,
                    accent_color=0,
                )
            },
        )

    @staticmethod
    async def get_user(credentials: OAuth2Credentials) -> DiscordOAuthUser:
        user_info = await get_current_user(credentials)
        return user_info

    async def run(
        self, input_data: Input, *, credentials: OAuth2Credentials, **kwargs
    ) -> BlockOutput:
        try:
            result = await self.get_user(credentials)

            # Yield each output field
            yield "user_id", result.user_id
            yield "username", result.username
            yield "avatar_url", result.avatar_url

            # Handle banner URL if banner hash exists
            if result.banner:
                banner_url = f"https://cdn.discordapp.com/banners/{result.user_id}/{result.banner}.png"
                yield "banner_url", banner_url
            else:
                yield "banner_url", ""

            yield "accent_color", result.accent_color or 0

        except Exception as e:
            raise ValueError(f"Failed to get Discord user info: {e}")
