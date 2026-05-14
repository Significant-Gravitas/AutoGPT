"""Public response models + sanitizer for shared chat sessions.

A shared chat link is fundamentally a public URL — anyone can read it.
Three classes of data therefore must NEVER leave the backend on this
path:

1. ``ChatSession.credentials`` — provider auth metadata.  Dropped wholesale.
2. ``ChatMessage.metadata`` — dispatcher payload (file_ids, model,
   permissions, request_arrival_at).  Dropped wholesale.
3. Tool-call arguments / function-call arguments shaped like secrets —
   keys matching :data:`_SECRET_KEY_HINTS`.  Redacted in place so the
   conversation still renders coherently, but the values never leak.

The sanitizer also strips server-injected context blocks from user
messages (memory_context / env_context / user_context) via the existing
:func:`strip_injected_context_for_display`.  That data is never owned
by the user and would confuse a public reader.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from backend.copilot.model import ChatMessage as ChatMessageDomain
from backend.copilot.model import ChatSessionInfo
from backend.copilot.service import strip_injected_context_for_display

# Substrings that mark a tool-call argument key as secret-shaped.  We
# redact rather than drop so the schema stays stable for the viewer.
_SECRET_KEY_HINTS = (
    "api_key",
    "apikey",
    "auth",
    "token",
    "secret",
    "credential",
    "cookie",
    "password",
    "passwd",
)
_REDACTED = "[redacted]"


class SharedChatLinkedExecution(BaseModel):
    """Drill-in pointer for an AgentGraphExecution shared alongside a chat."""

    execution_id: str
    graph_id: str
    graph_name: str | None
    share_token: str | None
    """``None`` when the linked execution was NOT opted-in at share time.

    The viewer can still render the inline tool-call snapshot but cannot
    deep-link into the full execution page."""


class SharedChatMessage(BaseModel):
    """Public-safe projection of a single chat message."""

    id: str
    role: str
    content: str | None
    name: str | None
    tool_call_id: str | None
    tool_calls: list[dict] | None
    function_call: dict | None
    sequence: int
    created_at: datetime


class SharedChatSession(BaseModel):
    """Public-safe projection of a chat session (header only)."""

    id: str
    title: str | None
    created_at: datetime
    updated_at: datetime
    linked_executions: list[SharedChatLinkedExecution]


class SharedChatMessagesPage(BaseModel):
    """Paginated message window for the public viewer."""

    messages: list[SharedChatMessage]
    has_more: bool
    oldest_sequence: int | None


def sanitize_chat_session(
    session: ChatSessionInfo,
    *,
    linked_executions: list[SharedChatLinkedExecution],
) -> SharedChatSession:
    """Project a session for public consumption.

    Drops credentials and dispatcher metadata; keeps only display fields.
    """
    return SharedChatSession(
        id=session.session_id,
        title=session.title,
        created_at=session.started_at,
        updated_at=session.updated_at,
        linked_executions=linked_executions,
    )


def sanitize_chat_message(message: ChatMessageDomain) -> SharedChatMessage:
    """Project a message for public consumption.

    - Strips injected context blocks from ``role=user`` string content.
    - Redacts secret-shaped keys inside ``tool_calls`` and ``function_call``.
    - Drops the per-row dispatcher metadata entirely.
    - Drops ``refusal`` (model-internal signal, not user content).

    Plain ``content`` is intentionally **not** key-redacted.  Tool-call
    arguments are structured data with stable secret-shaped keys
    (``api_key``, ``auth_token``) that we can pattern-match safely, but
    free-form chat content has no such structure -- any regex pass would
    either miss most real cases or false-positive on legitimate text.
    The share modal's warning banner ("don't share if it contains
    secrets you pasted") makes plain-content exposure an explicit
    user-opt-in decision.
    """
    content = message.content
    if message.role == "user" and isinstance(content, str) and content:
        content = strip_injected_context_for_display(content)

    return SharedChatMessage(
        id=message.id or "",
        role=message.role,
        content=content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=_redact_tool_calls(message.tool_calls),
        function_call=_redact_secret_keys(message.function_call),
        sequence=message.sequence or 0,
        created_at=message.created_at or datetime.fromtimestamp(0),
    )


def _redact_tool_calls(tool_calls: list[dict] | None) -> list[dict] | None:
    if tool_calls is None:
        return None
    return [_redact_secret_keys(tc) for tc in tool_calls]


def _redact_secret_keys(value: Any) -> Any:
    """Recursively walk *value* and redact secret-shaped subtrees.

    Returns a new structure; never mutates the input.  When a dict key
    matches a secret-shaped hint, the ENTIRE subtree under that key is
    replaced with the ``_REDACTED`` sentinel — including lists and
    nested dicts.  Pre-fix this only redacted string leaves at the
    immediate level, so payloads like ``{"api_keys": ["sk-1", "sk-2"]}``
    or ``{"auth": {"bearer": "tok"}}`` leaked through because the
    inner leaves had no secret-shaped key wrapping them.  Lose the
    (rare, low-value) ability to surface bool/int metadata under a
    secret-shaped key — the sanitizer is the load-bearing security
    boundary for the public viewer.
    """
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _is_secret_key(k) else _redact_secret_keys(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact_secret_keys(item) for item in value]
    return value


def _is_secret_key(key: str) -> bool:
    lower = key.lower()
    return any(hint in lower for hint in _SECRET_KEY_HINTS)
