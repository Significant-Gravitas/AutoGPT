"""
Centralized service client helpers with thread caching.
"""

from typing import TYPE_CHECKING

from autogpt_libs.utils.cache import thread_cached

if TYPE_CHECKING:
    from backend.data.execution import (
        AsyncRedisExecutionEventBus,
        RedisExecutionEventBus,
    )
    from backend.data.rabbitmq import AsyncRabbitMQ, SyncRabbitMQ
    from backend.executor import DatabaseManagerAsyncClient, DatabaseManagerClient
    from backend.executor.scheduler import SchedulerClient
    from backend.integrations.credentials_store import IntegrationCredentialsStore
    from backend.notifications.notifications import NotificationManagerClient


@thread_cached
def get_database_manager_client() -> "DatabaseManagerClient":
    """Get a thread-cached DatabaseManagerClient with request retry enabled."""
    from backend.executor import DatabaseManagerClient
    from backend.util.service import get_service_client

    return get_service_client(DatabaseManagerClient, request_retry=True)


@thread_cached
def get_database_manager_async_client() -> "DatabaseManagerAsyncClient":
    """Get a thread-cached DatabaseManagerAsyncClient with request retry enabled."""
    from backend.executor import DatabaseManagerAsyncClient
    from backend.util.service import get_service_client

    return get_service_client(DatabaseManagerAsyncClient, request_retry=True)


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


# ============ Integration Credentials Store ============ #


@thread_cached
def get_integration_credentials_store() -> "IntegrationCredentialsStore":
    """Get a thread-cached IntegrationCredentialsStore."""
    from backend.integrations.credentials_store import IntegrationCredentialsStore

    return IntegrationCredentialsStore()


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
