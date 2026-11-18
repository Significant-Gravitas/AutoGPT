import os
import re
import sys
from collections import deque
from enum import Enum
from types import TracebackType
from typing import Any

from pydantic import SecretStr
from sentry_sdk._init_implementation import init as _sentry_init
from sentry_sdk.api import capture_exception as _sentry_capture_exception
from sentry_sdk.api import flush as _sentry_flush
from sentry_sdk.consts import SPANDATA
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.scrubber import DEFAULT_DENYLIST, DEFAULT_PII_DENYLIST

try:
    from sentry_sdk.integrations.anthropic import AnthropicIntegration
except ImportError:
    AnthropicIntegration = None  # type: ignore[assignment,misc]

from backend.util.exceptions import get_execution_failure_reason
from backend.util.request_credentials import longest_first, request_credentials
from backend.util.security import SENSITIVE_FIELD_NAMES
from backend.util.settings import BehaveAs, Settings
from backend.util.tracing import traces_sampler
from backend.util.url_redaction import (
    MIN_CREDENTIAL_LENGTH,
    REDACTED,
    credentials_in_url,
    is_credential_header,
    redact_query_string,
    redact_url,
    redact_urls_in_text,
)

settings = Settings()


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

# Pika reconnect noise: `conn_retry` already handles these. Drop only the
# four known signatures so genuine pika ERRORs (auth, declare failures) still
# reach Sentry. (AUTOGPT-SERVER-6JC/6JD/6JE/6JF.)
_PIKA_RECONNECT_LOGGERS = {
    "pika.adapters.utils.io_services_utils",
    "pika.adapters.blocking_connection",
    "pika.adapters.base_connection",
}
_PIKA_RECONNECT_SIGNATURES = (
    "streamlosterror",
    "transport indicated eof",
    "socket eof",
    "connection_lost",
)

_SENTRY_SENSITIVE_FIELDS = sorted(
    set(DEFAULT_DENYLIST)
    | SENSITIVE_FIELD_NAMES
    | {
        "anthropic_auth_token",
        "bot_token",
        "client_secret",
        "codex_access_token",
        "codex_api_key",
        "device_code",
        "id_token",
        "login_code",
        "openai_api_key",
        "plain_text_key",
        "plaintext_key",
        "plaintext_secret",
        "plaintext_token",
        "provider_state",
        "secret_token",
        "secrets",
        "state_token",
        "user_code",
        "verification_url",
        # The credentials a route takes in its path, under their own names and
        # the database's.
        "share_token",
        "shareToken",
        "webhook_id",
        "webhookId",
    }
)
# A user is known to Sentry by their id only, so their email and IP address are
# filtered wherever else they turn up, such as in extras or a frame's locals.
# Exact names only: keys like ``email_verified`` stay readable.
_SENTRY_PERSONAL_FIELDS = ("email", "user_email", "recipient", "ip", "ip_address")
# Matched case-insensitively against every key in the event, as sentry's
# EventScrubber does (with its PII list, since send_default_pii is off).
_SENTRY_SENSITIVE_KEYS = frozenset(
    field.lower()
    for field in (
        *_SENTRY_SENSITIVE_FIELDS,
        *DEFAULT_PII_DENYLIST,
        *_SENTRY_PERSONAL_FIELDS,
    )
)
# The names a secret is assigned to, as a key or keyword argument; "_" in one
# stands for "_", "-" or nothing.
_SECRET_NAMES = (
    "access_token",
    "refresh_token",
    "id_token",
    "state_token",
    "bot_token",
    "share_token",
    "secret_token",
    "api_key",
    "authorization",
    "client_secret",
    "password",
    "provider_state",
    "secret",
    "device_code",
    "login_code",
    "user_code",
    "verification_url",
    "webhook_id",
    "plain_text_key",
    "plain_text_secret",
    "plain_text_token",
)
# Letters outside ASCII that IGNORECASE matches to an ASCII one.
_CASE_FOLDS = {"i": "İı", "k": "\u212a", "s": "ſ"}
_SECRET_NAME_STARTS = "".join(
    sorted(
        {name[0] for name in _SECRET_NAMES}
        | {name[0].upper() for name in _SECRET_NAMES}
        | {fold for name in _SECRET_NAMES for fold in _CASE_FOLDS.get(name[0], "")}
    )
)
_EMBEDDED_SECRET_ASSIGNMENT = re.compile(
    # A search starts only at one of the names' first letters, matched with
    # case, which the engine finds far faster than the names without it.
    f"(?=[{_SECRET_NAME_STARTS}])"
    + "(?i:"
    + "|".join(name.replace("_", "[_-]?") for name in _SECRET_NAMES)
    + ")"
    # An unset one, as a model's repr shows each of its fields, holds nothing.
    + r"(?:\\?[\"']?)\s*[:=](?!\s*(?:None|null|''|\"\")(?![\w'\"]))"
)
_TOKEN_SHAPED_VALUE = re.compile(
    r"(?i)(?:\b(?:bearer|basic)\s+[a-z0-9._~+/=-]{8,}|"
    # Matches what \beyJ... matched, entered once from the start of each base64url run.
    r"(?<![a-z0-9_-])(?>[a-z0-9_-]*?\beyJ)"
    r"[a-z0-9_-]{5,}\.[a-z0-9_-]{5,}\.[a-z0-9_-]{5,}\b|"
    r"\b(?:sk-(?:proj-)?|gh[pousr]_|github_pat_|xox[baprs]-)[a-z0-9_-]{8,})"
)
_FILTERED_VALUE = "[Filtered]"
# Three bounds in the order the SDK applies them: ``custom_repr`` caps a frame local
# before it is repr()'d, ``max_value_length`` then cuts every string in the event, and
# the ``before_send`` scrubber searches what is left.
_MAX_LOCAL_REPR = 1024
# The SDK cuts every string in an event to this length before ``before_send``.
_SENTRY_MAX_VALUE_LENGTH = 8 * 1024
# Characters of text the scrubber searches per event; strings past it are filtered.
_SENTRY_SCRUB_BUDGET = 512 * 1024
# Sentry lists breadcrumbs, chained exceptions and frames oldest or outermost first;
# the scrubber walks them from the end so the budget goes to the newest.
_OLDEST_FIRST_LISTS = frozenset({"values", "frames"})
# The HTTP client integrations record each outgoing request's query and
# fragment as sent, an API key in ``?key=`` included, and an ASGI scope among a
# frame's variables holds the raw query of the request.
_QUERY_KEYS = frozenset({SPANDATA.HTTP_QUERY, SPANDATA.HTTP_FRAGMENT, "query_string"})
# The SDK's own IDs, which hold a trace together, are never searched or charged
# to the budget, so a transaction past it keeps its shape.
_TRACE_ID_KEYS = frozenset({"event_id", "trace_id", "span_id", "parent_span_id"})


def _scrub_sentry_event(event: dict, *, secret_shapes: bool = True) -> None:
    """Scrub sensitive keys and secret-shaped values from an outgoing event.

    ``before_send`` sees the event after the SDK has serialized it, and the
    transport then ``json.dumps`` it as is. So every replacement must be plain
    JSON: sentry's EventScrubber writes ``AnnotatedValue`` objects, and one of
    those here made the SDK drop the whole event as an internal error. ``_meta``
    is the SDK's own annotation tree and is left alone.

    It searches at most ``_SENTRY_SCRUB_BUDGET`` characters per event and filters
    any string beyond that rather than send it unsearched; it walks breadth-first,
    newest breadcrumb and innermost frame first, and the breadcrumbs get only what
    the rest of the event left.

    An event raised in a request carries the variables of every frame, the
    route's included, so each credential the request held in its URL or headers
    is masked wherever it turns up, in a key as in a value, and so are the
    credentials in any URL the event quotes. ``secret_shapes=False`` skips the
    search for secret-shaped values, the costliest part.
    """
    # Read from the request's URL before it is masked.
    credentials = _known_credentials(event)
    _redact_request_urls(event)
    keys = [key for key in event if key not in ("_meta", "breadcrumbs")]
    budget = _scrub_breadth_first(
        [(event, key) for key in keys],
        _SENTRY_SCRUB_BUDGET,
        credentials,
        secret_shapes,
    )
    if "breadcrumbs" in event:
        _scrub_breadth_first(
            [(event, "breadcrumbs")], budget, credentials, secret_shapes
        )


def _scrub_breadth_first(
    roots: list[tuple[dict[Any, Any], Any]],
    budget: int,
    credentials: tuple[str, ...] = (),
    secret_shapes: bool = True,
) -> int:
    """Scrub the values under ``roots`` and return the budget left over."""
    queue: deque[tuple[dict[Any, Any] | list[Any], Any]] = deque(roots)
    while queue:
        parent, key = queue.popleft()
        value = parent[key]
        if key == "headers":
            _mask_headers(value)
        if isinstance(value, dict):
            value, budget = _scrub_keys(parent, key, budget, credentials, secret_shapes)
            for nested_key in value:
                if (
                    isinstance(nested_key, str)
                    and nested_key.lower() in _SENTRY_SENSITIVE_KEYS
                ):
                    value[nested_key] = _FILTERED_VALUE
                elif nested_key not in _TRACE_ID_KEYS:
                    queue.append((value, nested_key))
        elif isinstance(value, list):
            indices = range(len(value))
            if key in _OLDEST_FIRST_LISTS:
                indices = indices[::-1]
            queue.extend((value, index) for index in indices)
        elif isinstance(value, str):
            if key in _QUERY_KEYS:
                value = redact_query_string(value)
            if len(value) > budget:
                parent[key] = _FILTERED_VALUE
                continue
            budget -= len(value)
            parent[key] = _scrub_text(value, credentials, secret_shapes)
    return budget


def _scrub_text(
    text: str, credentials: tuple[str, ...], secret_shapes: bool = True
) -> str:
    """``text``, or ``[Filtered]`` if it holds a secret-shaped value, with each of
    ``credentials`` and the credentials in any URL in it masked."""
    if secret_shapes and (
        _EMBEDDED_SECRET_ASSIGNMENT.search(text) or _TOKEN_SHAPED_VALUE.search(text)
    ):
        return _FILTERED_VALUE
    for credential in credentials:
        if credential in text:
            text = text.replace(credential, REDACTED)
    return redact_urls_in_text(text)


def _scrub_keys(
    parent: dict[Any, Any] | list[Any],
    key: Any,
    budget: int,
    credentials: tuple[str, ...],
    secret_shapes: bool,
) -> tuple[dict[Any, Any], int]:
    """Scrub the keys of the mapping at ``parent[key]`` as text, renaming each
    that holds a secret, such as a mapping from a token to what it grants. A key
    shorter than any credential is a name, and is kept; so is a key past the
    budget, as the event's shape depends on its keys."""
    value: dict[Any, Any] = parent[key]
    renames: dict[str, str] = {}
    for nested_key in value:
        if (
            isinstance(nested_key, str)
            and MIN_CREDENTIAL_LENGTH <= len(nested_key) <= budget
        ):
            budget -= len(nested_key)
            scrubbed = _scrub_text(nested_key, credentials, secret_shapes)
            if scrubbed != nested_key:
                renames[nested_key] = scrubbed
    if not renames:
        return value, budget
    renamed: dict[Any, Any] = {}
    for nested_key, nested in value.items():
        new_key = renames.get(nested_key, nested_key)
        while nested_key in renames and (new_key in renamed or new_key in value):
            new_key += "*"
        renamed[new_key] = nested
    parent[key] = renamed
    return renamed, budget


def _mask_headers(headers: object) -> None:
    """Mask every credential header, whatever it is called: the SDK knows a
    fixed list of names, which misses a webhook provider's secret token such as
    Telegram's. Headers come as a mapping, or, in an ASGI scope among a frame's
    variables, as name and value pairs, the names reprs of bytes, such as
    ``["b'cookie'", "b'session=...'"]``."""
    if isinstance(headers, dict):
        for name in headers:
            if isinstance(name, str) and is_credential_header(name):
                headers[name] = _FILTERED_VALUE
    elif isinstance(headers, list):
        for pair in headers:
            if (
                isinstance(pair, list)
                and len(pair) == 2
                and isinstance(pair[0], str)
                and is_credential_header(pair[0])
            ):
                pair[1] = _FILTERED_VALUE


def _known_credentials(event: dict) -> tuple[str, ...]:
    """The credentials of the request the event was raised in: those its app
    kept from the URL and headers, and those in the URL the event reports."""
    request = event.get("request") or {}
    url, query = request.get("url"), request.get("query_string")
    found = set(request_credentials())
    if isinstance(url, str):
        found |= credentials_in_url(f"{url}?{query}" if isinstance(query, str) else url)
    return longest_first(found)


def _redact_request_urls(event: dict) -> None:
    """Mask credentials in the request URL the ASGI integration attaches.

    EventScrubber only scrubs a request's headers, cookies and body, so the
    query string, which for the WebSocket holds the user's session JWT, and
    the path secrets in ``url`` would otherwise be sent with every event.
    """
    request = event.get("request") or {}
    if isinstance(request.get("url"), str):
        request["url"] = redact_url(request["url"])
    # The SDK has already decoded the query, so a %26 inside a value reads as a
    # separator and the rest of the value as a key. None of it can be kept.
    if request.get("query_string"):
        request["query_string"] = REDACTED
    # A request no route matched is named after its raw URL.
    transaction = event.get("transaction")
    source = (event.get("transaction_info") or {}).get("source")
    if source == "url" and isinstance(transaction, str):
        event["transaction"] = redact_url(transaction)


# FalkorDB (Graphiti CoPilot memory) connection-teardown noise. graphiti-core's
# ``execute_query`` logs ERROR + re-raises whenever a query races the closing of
# its redis.asyncio connection — the fire-and-forget cache-eviction close or a
# per-request ``driver.close()`` in a finally block. Chat degrades gracefully
# (memory callers swallow it), so these are benign. Drop only the two teardown
# signatures, scoped to the graphiti driver (by logger for log events, by
# traceback for exceptions) so genuine query errors (Cypher typos, missing
# graphs) — and unrelated main-redis outages — still reach Sentry. (SENTRY-1387.)
_FALKORDB_DRIVER_LOGGER = "graphiti_core.driver.falkordb_driver"
_FALKORDB_TEARDOWN_SIGNATURES = (
    "buffer is closed",
    "connection closed by server",
)


def _raised_from_graphiti_falkordb(exc_tb: TracebackType | None) -> bool:
    """True if the graphiti FalkorDB driver module appears in the traceback.

    graphiti-core's ``execute_query`` re-raises from that module, so its frame
    is on the traceback of a teardown error originating there. Scoping to it
    keeps unrelated redis exceptions (e.g. the platform's main redis) reaching
    Sentry even when they share the ``connection closed by server`` message.
    """
    while exc_tb is not None:
        if exc_tb.tb_frame.f_globals.get("__name__") == _FALKORDB_DRIVER_LOGGER:
            return True
        exc_tb = exc_tb.tb_next
    return False


def _before_send(event, hint):
    """Filter out expected/transient errors from Sentry to reduce noise."""
    if "exc_info" in hint:
        exc_type, exc_value, exc_tb = hint["exc_info"]
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

        # FalkorDB connection-teardown races (eviction/shutdown) — benign, but
        # only when raised from the graphiti driver, so an unrelated main-redis
        # outage sharing the message still reaches Sentry.
        if any(
            sig in exc_msg for sig in _FALKORDB_TEARDOWN_SIGNATURES
        ) and _raised_from_graphiti_falkordb(exc_tb):
            return None

        # User-caused credential/auth/integration errors — not platform bugs
        if any(kw in exc_msg for kw in _USER_AUTH_KEYWORDS):
            return None

        # Expected business logic — exhausted wallet, or a plan that does
        # not include the feature. Neither is a platform bug.
        if get_execution_failure_reason(exc_value) is not None:
            return None

        # Expected security check — blocked IP access
        if "access to blocked or private ip" in exc_msg:
            return None

        # Discord bot token misconfiguration — not a platform error
        if "improper token has been passed" in exc_msg or (
            exc_type and exc_type.__name__ == "Forbidden" and "50001" in exc_msg
        ):
            return None

        # Prisma UniqueViolationError is always handled (FastAPI returns 400).
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
    logger_name = event.get("logger")
    if logger_name in _PIKA_RECONNECT_LOGGERS and log_msg:
        if any(sig in log_msg.lower() for sig in _PIKA_RECONNECT_SIGNATURES):
            return None
    if logger_name == _FALKORDB_DRIVER_LOGGER and log_msg:
        if any(sig in log_msg.lower() for sig in _FALKORDB_TEARDOWN_SIGNATURES):
            return None
    if logger_name and log_msg:
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

    _strip_user_pii(event)
    _strip_ip_headers(event)
    _scrub_sentry_event(event)
    return event


def _before_send_transaction(event, hint):
    """Transactions skip ``before_send`` but carry the same request data, and
    every WebSocket connection is one. Their spans record what the request did
    with its credentials too: each outgoing request's URL and query, and each
    Redis command's key.

    The search for secret-shaped values is left out. Its finds are in error
    messages and variables, which a transaction doesn't carry, and on every
    request's transaction it took longer than the SDK's own serializing of it.
    """
    _strip_user_pii(event)
    _strip_ip_headers(event)
    _scrub_sentry_event(event, secret_shapes=False)
    return event


# Sentry knows a user by their id only: never their email, name or IP
# address, whichever code put them on the scope.
_SENTRY_USER_FIELDS = frozenset({"id"})


def _strip_user_pii(event: dict) -> None:
    user = event.get("user")
    if user is None:
        return
    kept = {
        field: value for field, value in user.items() if field in _SENTRY_USER_FIELDS
    }
    if kept:
        event["user"] = kept
    else:
        del event["user"]


# Request headers that carry the visitor's IP address or the location derived
# from it. The backends sit behind Cloudflare, which adds ``Cf-Connecting-Ip``
# and its ``Cf-Ip*`` location headers to every request; sentry-sdk only drops
# ``X-Forwarded-For`` and ``X-Real-Ip`` with PII off.
_IP_HEADER_MARKERS = (
    "forwarded",
    "proxied_for",
    "real_ip",
    "connecting_ip",
    "client_ip",
    "vercel_ip",
)


def _is_ip_header(name: str) -> bool:
    key = name.lower().replace("-", "_")
    return key.startswith("cf_ip") or any(m in key for m in _IP_HEADER_MARKERS)


def _strip_ip_headers(event: dict) -> None:
    request = event.get("request")
    if not isinstance(request, dict):
        return
    headers = request.get("headers")
    if not isinstance(headers, dict):
        return
    request["headers"] = {
        name: value for name, value in headers.items() if not _is_ip_header(name)
    }


def _sentry_environment() -> str:
    """The environment events are tagged with. A deployment that names itself
    (a PR preview, say) takes precedence over the one computed from the app
    environment and behaviour mode. A blank name counts as none."""
    config = settings.config
    return (
        config.sentry_environment.strip()
        or f"app:{config.app_env.value}-behave:{config.behave_as.value}"
    )


def sentry_init():
    if _running_under_pytest():
        return

    sentry_dsn = settings.secrets.sentry_dsn
    optional_integrations = (
        [AnthropicIntegration(include_prompts=False)]
        if AnthropicIntegration is not None
        else []
    )
    _sentry_init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        traces_sampler=traces_sampler,
        # Traces reach our own services only, through the RPC client's headers
        # (backend.util.service); outbound calls to third parties, such as a
        # block's request to a user-supplied URL, carry no Sentry headers.
        trace_propagation_targets=[],
        # Continuous profiling: one sample buffer per process, however many
        # transactions are in flight, profiling while any traced one runs.
        profile_session_sample_rate=1.0,
        profile_lifecycle="trace",
        release=settings.config.sentry_release.strip() or None,
        environment=_sentry_environment(),
        # The SDK default, pinned: no IP address, request headers with
        # personal data or cookies.
        send_default_pii=False,
        max_value_length=_SENTRY_MAX_VALUE_LENGTH,
        before_send=_before_send,
        before_send_transaction=_before_send_transaction,
        custom_repr=_bounded_repr,
        integrations=[
            AsyncioIntegration(),
            LoggingIntegration(),
        ]
        + optional_integrations,
    )


def _bounded_repr(value: object) -> str | None:
    """The SDK repr()s frame locals in full while holding the GIL, which takes seconds
    and gigabytes when a traceback holds a large upload."""
    if isinstance(value, str) and len(value) > _MAX_LOCAL_REPR:
        head = _masked_head(value, request_credentials())
    elif isinstance(value, (bytes, bytearray)) and len(value) > _MAX_LOCAL_REPR:
        head = value[:_MAX_LOCAL_REPR]
    else:
        return None
    return f"{head!r}... ({len(value):,} long)"


def _masked_head(value: str, credentials: tuple[str, ...]) -> str:
    """The first ``_MAX_LOCAL_REPR`` characters of ``value`` with ``credentials``
    masked, so none is cut in two and half of it kept. Only as much of ``value``
    is read as the masked head needs."""
    if not credentials:
        return value[:_MAX_LOCAL_REPR]
    # Longest first: a credential that starts in the head ends within this.
    needed = _MAX_LOCAL_REPR + len(credentials[0])
    end = needed
    while True:
        head = value[:end]
        for credential in credentials:
            head = head.replace(credential, REDACTED)
        if len(head) >= needed or end >= len(value):
            return head[:_MAX_LOCAL_REPR]
        end += needed - len(head)


def _running_under_pytest() -> bool:
    """sentry_init() runs at import time (AppProcess class body), before pytest
    sets PYTEST_CURRENT_TEST; the env var covers spawned service subprocesses,
    which do not inherit sys.modules. Self-hosted instances match neither."""
    return "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ


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
