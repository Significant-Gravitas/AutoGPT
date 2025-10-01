import logging
from enum import Enum

import sentry_sdk
from pydantic import SecretStr
from sentry_sdk.integrations.anthropic import AnthropicIntegration
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from backend.util.settings import Settings

settings = Settings()


class DiscordChannel(str, Enum):
    PLATFORM = "platform"  # For platform/system alerts
    PRODUCT = "product"  # For product alerts (low balance, zero balance, etc.)


def sentry_init():
    sentry_dsn = settings.secrets.sentry_dsn
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
        environment=f"app:{settings.config.app_env.value}-behave:{settings.config.behave_as.value}",
        _experiments={"enable_logs": True},
        integrations=[
            AsyncioIntegration(),
            LoggingIntegration(sentry_logs_level=logging.INFO),
            AnthropicIntegration(
                include_prompts=False,
            ),
        ],
    )


def sentry_capture_error(error: BaseException):
    sentry_sdk.capture_exception(error)
    sentry_sdk.flush()


async def discord_send_alert(
    content: str, channel: DiscordChannel = DiscordChannel.PLATFORM
):
    from backend.blocks.discord.bot_blocks import SendDiscordMessageBlock
    from backend.data.model import APIKeyCredentials, CredentialsMetaInput, ProviderName

    creds = APIKeyCredentials(
        provider="discord",
        api_key=SecretStr(settings.secrets.discord_bot_token),
        title="Provide Discord Bot Token for the platform alert",
        expires_at=None,
    )

    # Select channel based on enum
    if channel == DiscordChannel.PLATFORM:
        channel_name = settings.config.platform_alert_discord_channel
    elif channel == DiscordChannel.PRODUCT:
        channel_name = settings.config.product_alert_discord_channel
    else:
        channel_name = settings.config.platform_alert_discord_channel

    return await SendDiscordMessageBlock().run_once(
        SendDiscordMessageBlock.Input(
            credentials=CredentialsMetaInput(
                id=creds.id,
                title=creds.title,
                type=creds.type,
                provider=ProviderName.DISCORD,
            ),
            message_content=content,
            channel_name=channel_name,
        ),
        "status",
        credentials=creds,
    )
