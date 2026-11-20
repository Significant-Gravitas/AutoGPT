"""Public response models + sanitizer for shared chat sessions.

A shared chat link is fundamentally a public URL — anyone can read it.
Three classes of data therefore must NEVER leave the backend on this
path:

1. ``ChatSession.credentials`` — provider auth metadata.  Dropped wholesale.
2. ``ChatMessage.metadata`` — dispatcher payload (file_ids, model,
   permissions, request_arrival_at).  Dropped wholesale.
3. Tool-call arguments / function-call arguments shaped like secrets —
   keys matching ``_SECRET_KEY_HINTS`` in
   :mod:`backend.copilot.sharing.redaction`.  Redacted in place so the
   conversation still renders coherently, but the values never leak.

The sanitizer also strips server-injected context blocks from user
messages (memory_context / env_context / user_context) via the existing
:func:`strip_injected_context_for_display`.  That data is never owned
by the user and would confuse a public reader.
"""

from datetime import datetime

from pydantic import BaseModel

from backend.copilot.gate.held import HeldResult
from backend.copilot.model import ChatMessage as ChatMessageDomain
from backend.copilot.model import ChatSessionInfo
from backend.copilot.service import strip_injected_context_for_display
from backend.copilot.sharing.redaction import (
    redact_held_call_result,
    redact_tool_call,
    redact_tool_calls,
    redact_tool_output,
)


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
    shared_at: datetime | None
    """When the owner enabled sharing on this chat.  Surfaced so the
    public viewer can render "Shared {date}" against the share-enable
    moment instead of the chat creation date (which can be months
    older for long-lived chats)."""
    linked_executions: list[SharedChatLinkedExecution]


class SharedChatMessagesPage(BaseModel):
    """Paginated message window for the public viewer."""

    messages: list[SharedChatMessage]
    has_more: bool
    oldest_sequence: int | None


class ChatShareState(BaseModel):
    """Owner-facing share state for a chat session (not a public model).

    Powers the share modal: ``is_shared`` / ``share_token`` decide the
    share-vs-revoke mode, and the three counts drive the consent-disclosure
    block above the toggle so the owner sees exactly what they're about to
    expose.  The counts reflect live state — sharing is live, not
    snapshot-at-enable — so they ARE the numbers visible the moment the
    owner enables sharing.
    """

    is_shared: bool
    share_token: str | None
    auto_share_executions: bool = False
    message_count: int = 0
    linked_run_count: int = 0
    file_count: int = 0


def sanitize_chat_session(
    session: ChatSessionInfo,
    *,
    linked_executions: list[SharedChatLinkedExecution],
    shared_at: datetime | None = None,
) -> SharedChatSession:
    """Project a session for public consumption.

    Drops credentials and dispatcher metadata; keeps only display fields.
    """
    return SharedChatSession(
        id=session.session_id,
        title=session.title,
        created_at=session.started_at,
        updated_at=session.updated_at,
        shared_at=shared_at,
        linked_executions=linked_executions,
    )


def sanitize_chat_message(message: ChatMessageDomain) -> SharedChatMessage:
    """Project a message for public consumption.

    - Strips injected context blocks from ``role=user`` string content.
    - Redacts the tool output in a held call's late result, which the gate
      stores as a ``role=user`` row, as it would a ``role=tool`` result.  The
      row is known by its metadata or its opening tag; one not laid out as
      the gate writes it has its whole body withheld.
    - Redacts secret-shaped keys inside ``tool_calls`` and ``function_call``,
      including inside the JSON string each call's arguments are stored as.
    - Redacts secret-shaped keys inside ``role=tool`` JSON content (tool
      responses carry structured payloads like ``agent_output``,
      ``view_agent_output``, dry-run ``node_executions`` with block I/O —
      any of which may contain a secret-shaped field).  A result too large
      to keep inline is stored as a preview, or cut in the middle when its
      full copy can't be saved; neither can be checked by key, so neither
      is shown, and neither is what a read returns from that full copy or
      any other of Otto's working files.  Text that doesn't parse, in a
      result or a call, is withheld when it quotes a secret-shaped key before
      a colon (JSON lines, a dict's repr, a window of a JSON file); other
      non-JSON content (free-form strings) passes through unchanged, same
      posture as ``role=assistant``.
    - Drops the per-row dispatcher metadata entirely.
    - Drops ``refusal`` (model-internal signal, not user content).

    Plain ``content`` on ``role=user`` / ``role=assistant`` is intentionally
    **not** key-redacted.  Tool-call arguments are structured data with
    stable secret-shaped keys (``api_key``, ``auth_token``) that we can
    pattern-match safely, but free-form chat content has no such structure
    -- any regex pass would either miss most real cases or false-positive
    on legitimate text.  The share modal's warning banner ("don't share
    if it contains secrets you pasted") makes plain-content exposure an
    explicit user-opt-in decision.
    """
    content = message.content
    if message.role == "user" and isinstance(content, str) and content:
        content = redact_held_call_result(
            strip_injected_context_for_display(content),
            held=HeldResult.carried_by({"metadata": message.metadata}),
        )
    elif message.role == "tool" and isinstance(content, str) and content:
        content = redact_tool_output(content)

    return SharedChatMessage(
        id=message.id or "",
        role=message.role,
        content=content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=redact_tool_calls(message.tool_calls),
        function_call=redact_tool_call(message.function_call),
        sequence=message.sequence or 0,
        created_at=message.created_at or datetime.fromtimestamp(0),
    )
