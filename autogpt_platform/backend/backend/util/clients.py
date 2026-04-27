"""
Centralized service client helpers with thread caching.
"""

from typing import TYPE_CHECKING

from backend.util.cache import cached, thread_cached
from backend.util.settings import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

settings = Settings()

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from supabase import AClient, Client

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
def get_openai_client(*, prefer_openrouter: bool = False) -> "AsyncOpenAI | None":
    """
    Get a process-cached async OpenAI client.

    By default prefers openai_internal_api_key (direct OpenAI) and falls back
    to open_router_api_key via OpenRouter.

    When ``prefer_openrouter=True``, returns an OpenRouter client or None —
    does **not** fall back to direct OpenAI (which can't route non-OpenAI
    models like ``google/gemini-2.5-flash``).
    """
    from openai import AsyncOpenAI

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
