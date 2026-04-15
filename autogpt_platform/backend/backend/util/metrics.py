import logging
from enum import Enum

from pydantic import SecretStr
from sentry_sdk._init_implementation import init as _sentry_init
from sentry_sdk.api import capture_exception as _sentry_capture_exception
from sentry_sdk.api import flush as _sentry_flush
from sentry_sdk.integrations import DidNotEnable
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

try:
    from sentry_sdk.integrations.anthropic import AnthropicIntegration
except ImportError:
    AnthropicIntegration = None  # type: ignore[assignment,misc]

try:
    from sentry_sdk.integrations.launchdarkly import LaunchDarklyIntegration
except ImportError:
    LaunchDarklyIntegration = None  # type: ignore[assignment,misc]

from backend.util import feature_flag
from backend.util.settings import BehaveAs, Settings

settings = Settings()
logger = logging.getLogger(__name__)


class DiscordChannel(str, Enum):
    PLATFORM = "platform"  # For platform/system alerts
    PRODUCT = "product"  # For product alerts (low balance, zero balance, etc.)


_USER_AUTH_KEYWORDS = [
    "incorrect api key",
    "invalid x-api-key",
    "invalid api key",
    "missing authentication header",
    "invalid api token",
    "authentication_error",
    "bad credentials",
    "unauthorized",
    "insufficient authentication scopes",
    "http 401 error",
    "http 403 error",
]

_AMQP_KEYWORDS = [
    "amqpconnection",
    "amqpconnector",
    "connection_forced",
    "channelinvalidstateerror",
    "no active transport",
]

_AMQP_INDICATORS = ["aio_pika", "aiormq", "amqp", "pika", "rabbitmq"]


def _before_send(event, hint):
    """Filter out expected/transient errors from Sentry to reduce noise."""
    if "exc_info" in hint:
        exc_type, exc_value, _ = hint["exc_info"]
        exc_msg = str(exc_value).lower() if exc_value else ""

        # AMQP/RabbitMQ transient connection errors — expected during deploys
        if any(kw in exc_msg for kw in _AMQP_KEYWORDS):
            return None

        # "connection refused" only for AMQP-related exceptions (not other services)
        if "connection refused" in exc_msg:
            exc_module = getattr(exc_type, "__module__", "") or ""
            exc_name = getattr(exc_type, "__name__", "") or ""
            if any(
                ind in exc_module.lower() or ind in exc_name.lower()
                for ind in _AMQP_INDICATORS
            ) or any(kw in exc_msg for kw in _AMQP_INDICATORS):
                return None

        # User-caused credential/auth/integration errors — not platform bugs
        if any(kw in exc_msg for kw in _USER_AUTH_KEYWORDS):
            return None

        # Expected business logic — insufficient balance
        if "insufficient balance" in exc_msg or "no credits left" in exc_msg:
            return None

        # Expected security check — blocked IP access
        if "access to blocked or private ip" in exc_msg:
            return None

        # Discord bot token misconfiguration — not a platform error
        if "improper token has been passed" in exc_msg or (
            exc_type and exc_type.__name__ == "Forbidden" and "50001" in exc_msg
        ):
            return None

        # Prisma UniqueViolationError — always caught and handled in our codebase.
        # These arise from concurrent create operations racing on unique constraints
        # (workspace files, credits, library folders, store listings, chat messages).
        # Every call site has an except handler; the global FastAPI handler also
        # catches them and returns 400.  Safe to drop unconditionally.
        if exc_type and exc_type.__name__ == "UniqueViolationError":
            return None

        # Google metadata DNS errors — expected in non-GCP environments
        if (
            "metadata.google.internal" in exc_msg
            and settings.config.behave_as != BehaveAs.CLOUD
        ):
            return None

        # Inactive email recipients — expected for bounced addresses
        if "marked as inactive" in exc_msg or "inactive addresses" in exc_msg:
            return None

    # Also filter log-based events for known noisy messages.
    # Sentry's LoggingIntegration stores log messages under "logentry", not "message".
    logentry = event.get("logentry") or {}
    log_msg = (
        logentry.get("formatted") or logentry.get("message") or event.get("message")
    )
    if event.get("logger") and log_msg:
        msg = log_msg.lower()
        noisy_log_patterns = [
            "amqpconnection",
            "connection_forced",
            "unclosed client session",
            "unclosed connector",
        ]
        if any(p in msg for p in noisy_log_patterns):
            return None
        if "connection refused" in msg and any(ind in msg for ind in _AMQP_INDICATORS):
            return None
        # Same auth keywords — errors logged via logger.error() bypass exc_info
        if any(kw in msg for kw in _USER_AUTH_KEYWORDS):
            return None

    return event


def sentry_init():
    sentry_dsn = settings.secrets.sentry_dsn
    integrations = []
    if feature_flag.is_configured() and LaunchDarklyIntegration is not None:
        try:
            integrations.append(LaunchDarklyIntegration(feature_flag.get_client()))
        except DidNotEnable as e:
            logger.error(f"Error enabling LaunchDarklyIntegration for Sentry: {e}")
    optional_integrations = (
        [AnthropicIntegration(include_prompts=False)]
        if AnthropicIntegration is not None
        else []
    )
    _sentry_init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
        environment=f"app:{settings.config.app_env.value}-behave:{settings.config.behave_as.value}",
        before_send=_before_send,
        integrations=[
            AsyncioIntegration(),
            LoggingIntegration(),
        ]
        + optional_integrations
        + integrations,
    )


def sentry_capture_error(error: BaseException):
    _sentry_capture_exception(error)
    _sentry_flush()


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
