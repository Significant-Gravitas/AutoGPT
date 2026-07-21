"""
Centralized service client helpers with thread caching.
"""

from typing import TYPE_CHECKING

from backend.util.cache import cached, thread_cached
from backend.util.settings import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

settings = Settings()

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI
    from supabase import AClient, Client

    from backend.copilot.bot.app import CoPilotChatBridgeClient
    from backend.data.db_manager import (
        DatabaseManagerAsyncClient,
        DatabaseManagerClient,
    )
    from backend.data.execution import (
        AsyncRedisExecutionEventBus,
        RedisExecutionEventBus,
    )
    from backend.data.rabbitmq import AsyncRabbitMQ, SyncRabbitMQ
    from backend.executor.scheduler import SchedulerClient
    from backend.integrations.credentials_store import IntegrationCredentialsStore
    from backend.notifications.notifications import NotificationManagerClient
    from backend.platform_linking.manager import PlatformLinkingManagerClient


@thread_cached
def get_database_manager_client() -> "DatabaseManagerClient":
    """Get a thread-cached DatabaseManagerClient with request retry enabled."""
    from backend.data.db_manager import DatabaseManagerClient
    from backend.util.service import get_service_client

    return get_service_client(DatabaseManagerClient, request_retry=True)


@thread_cached
def get_database_manager_async_client(
    should_retry: bool = True,
) -> "DatabaseManagerAsyncClient":
    """Get a thread-cached DatabaseManagerAsyncClient with request retry enabled."""
    from backend.data.db_manager import DatabaseManagerAsyncClient
    from backend.util.service import get_service_client

    return get_service_client(DatabaseManagerAsyncClient, request_retry=should_retry)


@thread_cached
def get_scheduler_client() -> "SchedulerClient":
    """Get a thread-cached SchedulerClient."""
    from backend.executor.scheduler import SchedulerClient
    from backend.util.service import get_service_client

    return get_service_client(SchedulerClient)


@thread_cached
def get_notification_manager_client() -> "NotificationManagerClient":
    """Get a thread-cached NotificationManagerClient."""
    from backend.notifications.notifications import NotificationManagerClient
    from backend.util.service import get_service_client

    return get_service_client(NotificationManagerClient)


@thread_cached
def get_platform_linking_manager_client() -> "PlatformLinkingManagerClient":
    """Get a thread-cached PlatformLinkingManagerClient."""
    from backend.platform_linking.manager import PlatformLinkingManagerClient
    from backend.util.service import get_service_client

    return get_service_client(PlatformLinkingManagerClient)


@thread_cached
def get_copilot_chat_bridge_client() -> "CoPilotChatBridgeClient":
    """Get a thread-cached CoPilotChatBridgeClient."""
    from backend.copilot.bot.app import CoPilotChatBridgeClient
    from backend.util.service import get_service_client

    return get_service_client(CoPilotChatBridgeClient)


# ============ Execution Event Bus Helpers ============ #


@thread_cached
def get_execution_event_bus() -> "RedisExecutionEventBus":
    """Get a thread-cached RedisExecutionEventBus."""
    from backend.data.execution import RedisExecutionEventBus

    return RedisExecutionEventBus()


@thread_cached
def get_async_execution_event_bus() -> "AsyncRedisExecutionEventBus":
    """Get a thread-cached AsyncRedisExecutionEventBus."""
    from backend.data.execution import AsyncRedisExecutionEventBus

    return AsyncRedisExecutionEventBus()


# ============ Execution Queue Helpers ============ #


@thread_cached
def get_execution_queue() -> "SyncRabbitMQ":
    """Get a thread-cached SyncRabbitMQ execution queue client."""
    from backend.data.rabbitmq import SyncRabbitMQ
    from backend.executor.utils import create_execution_queue_config

    client = SyncRabbitMQ(create_execution_queue_config())
    client.connect()
    return client


@thread_cached
async def get_async_execution_queue() -> "AsyncRabbitMQ":
    """Get a thread-cached AsyncRabbitMQ execution queue client."""
    from backend.data.rabbitmq import AsyncRabbitMQ
    from backend.executor.utils import create_execution_queue_config

    client = AsyncRabbitMQ(create_execution_queue_config())
    await client.connect()
    return client


# ============ CoPilot Queue Helpers ============ #


@thread_cached
async def get_async_copilot_queue() -> "AsyncRabbitMQ":
    """Get a thread-cached AsyncRabbitMQ CoPilot queue client."""
    from backend.copilot.executor.utils import create_copilot_queue_config
    from backend.data.rabbitmq import AsyncRabbitMQ

    client = AsyncRabbitMQ(create_copilot_queue_config())
    await client.connect()
    return client


# ============ Integration Credentials Store ============ #


@thread_cached
def get_integration_credentials_store() -> "IntegrationCredentialsStore":
    """Get a thread-cached IntegrationCredentialsStore."""
    from backend.integrations.credentials_store import IntegrationCredentialsStore

    return IntegrationCredentialsStore()


# ============ Supabase Clients ============ #


@cached(ttl_seconds=3600)
def get_supabase() -> "Client":
    """Get a process-cached synchronous Supabase client instance."""
    from supabase import create_client

    return create_client(
        settings.secrets.supabase_url, settings.secrets.supabase_service_role_key
    )


@cached(ttl_seconds=3600)
async def get_async_supabase() -> "AClient":
    """Get a process-cached asynchronous Supabase client instance."""
    from supabase import create_async_client

    return await create_async_client(
        settings.secrets.supabase_url, settings.secrets.supabase_service_role_key
    )


# ============ OpenAI Client ============ #


@cached(ttl_seconds=3600)
def _get_local_openai_client() -> "AsyncOpenAI":
    """Build the local-transport OpenAI client.

    Factored out so both ``prefer_openrouter`` arms of ``get_openai_client``
    share a single cached instance under ``CHAT_USE_LOCAL=true`` — the cache
    keys on its own arglist (which is empty), not on the caller's
    ``prefer_openrouter`` kwarg, so we don't end up with two equivalent
    ``AsyncOpenAI`` instances pointed at the same endpoint.
    """
    from openai import AsyncOpenAI

    # Reuse the module-level ``ChatConfig`` singleton from
    # ``copilot.sdk.env`` rather than constructing a fresh one — that
    # config object already exists for the SDK env builder, runs the full
    # validator chain once at import time, and avoids the ``.env``-reread
    # cost on every cache miss here. Lazy-imported because
    # ``backend.copilot.config`` imports ``OPENROUTER_BASE_URL`` from this
    # module — a top-level import would create a cycle.
    from backend.copilot.sdk.env import config as chat_cfg

    return AsyncOpenAI(
        api_key=chat_cfg.api_key,
        base_url=chat_cfg.base_url,
        timeout=chat_cfg.local_request_timeout_s,
    )


@cached(ttl_seconds=3600)
def get_openai_client(*, prefer_openrouter: bool = False) -> "AsyncOpenAI | None":
    """
    Get a process-cached async OpenAI client.

    Resolution order:

    1. **Local transport** (``CHAT_USE_LOCAL=true``) wins unconditionally.
       Operators who opted into self-hosted shouldn't have platform helpers
       (dry-run simulator, prompt compression, marketplace embeddings, …)
       silently route to the cloud just because legacy cloud-key fallbacks
       happen to be present. Returns a client pointed at the same
       OpenAI-compatible endpoint AutoPilot uses, with the same generous
       request timeout — those helpers fire under the same hardware
       constraints (CPU-only Ollama is slow). ``prefer_openrouter`` is
       intentionally ignored here: the local client is the only sane
       routing target regardless of the caller's preference.
    2. ``prefer_openrouter=True`` → returns an OpenRouter client or None.
       Does **not** fall back to direct OpenAI (which can't route non-OpenAI
       models like ``google/gemini-2.5-flash``).
    3. Default → prefers ``openai_internal_api_key`` (direct OpenAI), falls
       back to ``open_router_api_key`` via OpenRouter.

    Returns ``None`` only when none of the above resolve — callers must
    handle that branch (see e.g. ``executor/simulator._run_simulation_llm``).
    Note that under ``CHAT_USE_LOCAL=true`` this never returns ``None``,
    so callers that previously used a ``None`` return as a "skip cloud-only
    feature" sentinel will instead exercise the local path — that is the
    intended behaviour for this PR.
    """
    from openai import AsyncOpenAI

    # Local transport takes precedence so a stray ``OPENAI_API_KEY`` set
    # for some other reason can't override an explicit ``CHAT_USE_LOCAL=true``.
    from backend.copilot.sdk.env import config as chat_cfg

    if chat_cfg.transport.name == "local":
        return _get_local_openai_client()

    openai_key = settings.secrets.openai_internal_api_key
    openrouter_key = settings.secrets.open_router_api_key

    if prefer_openrouter:
        if openrouter_key:
            return AsyncOpenAI(api_key=openrouter_key, base_url=OPENROUTER_BASE_URL)
        return None
    else:
        if openai_key:
            return AsyncOpenAI(api_key=openai_key)
        if openrouter_key:
            return AsyncOpenAI(api_key=openrouter_key, base_url=OPENROUTER_BASE_URL)
    return None


# ============ Anthropic Client ============ #


@cached(ttl_seconds=3600)
def get_anthropic_client() -> "AsyncAnthropic | None":
    """Get a process-cached async Anthropic client.

    Reads ``settings.secrets.anthropic_api_key`` (same env var the SDK
    blocks consume). Returns ``None`` when no key is configured — caller
    should fall back to OpenRouter-routed Anthropic via
    ``get_openai_client(prefer_openrouter=True)`` for sync calls, or
    skip the Anthropic-batch path entirely.

    Used by the dream pass + future batch-mode LLM callers that need
    direct Anthropic API access (batch API, prompt caching with
    cache_control, tool-use forced structured output).
    """
    from anthropic import AsyncAnthropic

    anthropic_key = settings.secrets.anthropic_api_key
    if anthropic_key:
        return AsyncAnthropic(api_key=anthropic_key)
    return None


def openrouter_helper_cost_provider() -> str:
    """Cost-log provider for a client from ``get_openai_client(prefer_openrouter=True)``.

    Mirrors that function's routing so the ``PlatformCostLog`` row names the
    endpoint the call physically hit, not the chat transport identity:

    - **Local transport** → ``"ollama"`` — the self-hosted endpoint
      ``get_openai_client`` returns unconditionally under ``CHAT_USE_LOCAL``.
    - **Every other transport** → ``"open_router"``. ``prefer_openrouter=True``
      only ever returns an OpenRouter client or ``None`` — it never falls back
      to direct Anthropic — so any call that actually billed went through
      OpenRouter, even under ``subscription`` / ``direct_anthropic`` transport
      with an ``OPEN_ROUTER_API_KEY`` present. Keying off
      ``transport.cost_log_provider`` here would mislabel those as ``"anthropic"``.
    """
    from backend.copilot.sdk.env import config as chat_cfg

    return "ollama" if chat_cfg.transport.name == "local" else "open_router"


# ============ Notification Queue Helpers ============ #


@thread_cached
def get_notification_queue() -> "SyncRabbitMQ":
    """Get a thread-cached SyncRabbitMQ notification queue client."""
    from backend.data.rabbitmq import SyncRabbitMQ
    from backend.notifications.notifications import create_notification_config

    client = SyncRabbitMQ(create_notification_config())
    client.connect()
    return client


@thread_cached
async def get_async_notification_queue() -> "AsyncRabbitMQ":
    """Get a thread-cached AsyncRabbitMQ notification queue client."""
    from backend.data.rabbitmq import AsyncRabbitMQ
    from backend.notifications.notifications import create_notification_config

    client = AsyncRabbitMQ(create_notification_config())
    await client.connect()
    return client
