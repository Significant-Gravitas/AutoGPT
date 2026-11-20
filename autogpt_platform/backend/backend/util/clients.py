"""
Centralized service client helpers with thread caching.
"""

import asyncio
from time import monotonic
from typing import TYPE_CHECKING, Awaitable, TypeVar

import httpx

from backend.util.cache import thread_cached
from backend.util.exceptions import SchedulerUnavailableError
from backend.util.per_loop_cache import per_loop_cached, same_loop_hooks
from backend.util.settings import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

settings = Settings()

T = TypeVar("T")

# A deploy leaves no scheduler for about 1-2.5 min, and at worst about 10: the
# 300s shutdown grace plus the 300s startup probe. An outage longer than that
# is not a deploy, and someone should hear about it.
SCHEDULER_SUSTAINED_OUTAGE_SECONDS = 600.0

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from openai import AsyncOpenAI

    from backend.copilot.bot.app import CoPilotChatBridgeClient
    from backend.data.db_manager import (
        DatabaseManagerAsyncClient,
        DatabaseManagerClient,
    )
    from backend.data.execution import (
        AsyncRedisExecutionEventBus,
        RedisExecutionEventBus,
    )
    from backend.data.rabbitmq import AsyncRabbitMQ
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
def get_scheduler_client(should_retry: bool = False) -> "SchedulerClient":
    """Get a thread-cached SchedulerClient.

    ``should_retry`` repeats a call that failed in transit, such as on a pooled
    connection the scheduler had just closed; only for calls safe to repeat.
    """
    from backend.executor.scheduler import SchedulerClient
    from backend.util.service import get_service_client

    return get_service_client(SchedulerClient, request_retry=should_retry)


async def read_schedules(read: Awaitable[T]) -> T:
    """Await a schedule list someone is waiting on, failing fast while the
    scheduler is out.

    The scheduler restarts on every deploy. A read it refuses, drops or leaves
    unanswered past ``schedule_list_timeout_seconds`` raises
    ``SchedulerUnavailableError`` (a 503), well inside the 30s frontend proxy
    and the 300s RPC timeout. A scheduler that answers with an error, its own
    timeout included, is not out, so that error passes through unchanged.
    """
    deadline = asyncio.timeout(settings.config.schedule_list_timeout_seconds)
    try:
        async with deadline:
            schedules = await read
    except httpx.TransportError as e:
        raise _scheduler_unavailable(e) from e
    except TimeoutError as e:
        if deadline.expired():
            raise _scheduler_unavailable(e) from e
        _scheduler_outage.answered()
        raise
    except Exception:
        _scheduler_outage.answered()
        raise
    _scheduler_outage.answered()
    return schedules


def _scheduler_unavailable(cause: Exception) -> SchedulerUnavailableError:
    return SchedulerUnavailableError(
        f"The scheduler did not answer: {type(cause).__name__} {cause}".rstrip(),
        outage_seconds=_scheduler_outage.failed(),
    )


class _SchedulerOutage:
    """This process's current run of failed schedule reads.

    Reads are rare, so failures more than ``SCHEDULER_SUSTAINED_OUTAGE_SECONDS``
    apart are separate outages, such as two deploys' gaps, as are failures
    with an answer between them.
    """

    def __init__(self) -> None:
        self.started: float | None = None
        self.last_failure: float | None = None

    def failed(self) -> float:
        """Count a failed read; return how long the current outage has lasted."""
        now = monotonic()
        started = self.started
        if (
            started is None
            or self.last_failure is None
            or now - self.last_failure > SCHEDULER_SUSTAINED_OUTAGE_SECONDS
        ):
            started = now
        self.started, self.last_failure = started, now
        return now - started

    def answered(self) -> None:
        self.started = self.last_failure = None


_scheduler_outage = _SchedulerOutage()


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


# ============ OpenAI Client ============ #


@per_loop_cached
def _get_local_openai_client() -> "AsyncOpenAI":
    """Build the local-transport OpenAI client.

    Factored out so both ``prefer_openrouter`` arms of ``get_openai_client``
    share a single cached instance per event loop under
    ``CHAT_USE_LOCAL=true`` — the cache keys on its own arglist (which is
    empty), not on the caller's ``prefer_openrouter`` kwarg, so we don't end
    up with two equivalent ``AsyncOpenAI`` instances pointed at the same
    endpoint.
    """
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient

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
        http_client=DefaultAsyncHttpxClient(
            event_hooks=same_loop_hooks("get_openai_client()")
        ),
    )


@per_loop_cached
def get_openai_client(*, prefer_openrouter: bool = False) -> "AsyncOpenAI | None":
    """
    Get the running event loop's cached async OpenAI client.

    Cached per loop, not per process: the executors run one loop per worker
    thread, and a client shared across loops sends a call to the provider
    twice (see ``backend/util/per_loop_cache.py``).

    Resolution order:

    1. **Local transport** (``CHAT_USE_LOCAL=true``) wins unconditionally.
       Operators who opted into self-hosted shouldn't have platform helpers
       (dry-run simulator, prompt compression, marketplace embeddings, …)
       silently route to the cloud just because legacy cloud-key fallbacks
       happen to be present. Returns a client pointed at the same
       OpenAI-compatible endpoint Otto uses, with the same generous
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
    # Local transport takes precedence so a stray ``OPENAI_API_KEY`` set
    # for some other reason can't override an explicit ``CHAT_USE_LOCAL=true``.
    from backend.copilot.sdk.env import config as chat_cfg

    if chat_cfg.transport.name == "local":
        return _get_local_openai_client()

    openai_key = settings.secrets.openai_internal_api_key
    openrouter_key = settings.secrets.open_router_api_key

    if prefer_openrouter:
        if openrouter_key:
            return _openai_client(openrouter_key, OPENROUTER_BASE_URL)
        return None
    else:
        if openai_key:
            return _openai_client(openai_key)
        if openrouter_key:
            return _openai_client(openrouter_key, OPENROUTER_BASE_URL)
    return None


def _openai_client(api_key: str, base_url: str | None = None) -> "AsyncOpenAI":
    """A ``get_openai_client`` client, which refuses to send from a second loop."""
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=DefaultAsyncHttpxClient(
            event_hooks=same_loop_hooks("get_openai_client()")
        ),
    )


# ============ Anthropic Client ============ #


@per_loop_cached
def get_anthropic_client() -> "AsyncAnthropic | None":
    """Get the running event loop's cached async Anthropic client.

    Reads ``settings.secrets.anthropic_api_key`` (same env var the SDK
    blocks consume). Returns ``None`` when no key is configured — caller
    should fall back to OpenRouter-routed Anthropic via
    ``get_openai_client(prefer_openrouter=True)`` for sync calls, or
    skip the Anthropic-batch path entirely.

    Used by the dream pass + future batch-mode LLM callers that need
    direct Anthropic API access (batch API, prompt caching with
    cache_control, tool-use forced structured output).
    """
    from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

    anthropic_key = settings.secrets.anthropic_api_key
    if anthropic_key:
        return AsyncAnthropic(
            api_key=anthropic_key,
            http_client=DefaultAsyncHttpxClient(
                event_hooks=same_loop_hooks("get_anthropic_client()")
            ),
        )
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
async def get_async_notification_queue() -> "AsyncRabbitMQ":
    """Get a thread-cached AsyncRabbitMQ notification queue client."""
    from backend.data.rabbitmq import AsyncRabbitMQ
    from backend.notifications.queue import create_notification_config

    client = AsyncRabbitMQ(create_notification_config())
    await client.connect()
    return client
