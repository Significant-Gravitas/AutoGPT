import asyncio
import logging

import sentry_sdk
from pydantic import SecretStr
from sentry_sdk.integrations.anthropic import AnthropicIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from backend.util.settings import Settings


def sentry_init():
    sentry_dsn = Settings().secrets.sentry_dsn
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
        environment=f"app:{Settings().config.app_env.value}-behave:{Settings().config.behave_as.value}",
        _experiments={"enable_logs": True},
        integrations=[
            LoggingIntegration(sentry_logs_level=logging.INFO),
            AnthropicIntegration(
                include_prompts=False,
            ),
        ],
    )


def sentry_capture_error(error: Exception):
    sentry_sdk.capture_exception(error)
    sentry_sdk.flush()


def discord_send_alert(content: str):
    from backend.blocks.discord import SendDiscordMessageBlock
    from backend.data.model import APIKeyCredentials, CredentialsMetaInput, ProviderName
    from backend.util.settings import Settings

    settings = Settings()
    creds = APIKeyCredentials(
        provider="discord",
        api_key=SecretStr(settings.secrets.discord_bot_token),
        title="Provide Discord Bot Token for the platform alert",
        expires_at=None,
    )

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return SendDiscordMessageBlock().run_once(
        SendDiscordMessageBlock.Input(
            credentials=CredentialsMetaInput(
                id=creds.id,
                title=creds.title,
                type=creds.type,
                provider=ProviderName.DISCORD,
            ),
            message_content=content,
            channel_name=settings.config.platform_alert_discord_channel,
        ),
        "status",
        credentials=creds,
    )
