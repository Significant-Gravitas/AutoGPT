"""Abstract base for platform adapters.

``PlatformAdapter`` is the outbound contract the platform-agnostic core handler
(``handler.py``) speaks through — it never names a platform. Concrete adapters
extend one of two subtypes depending on how inbound events arrive:
``SocketAdapter`` (owns a long-lived connection — Discord Gateway, Slack Socket
Mode) or ``WebhookAdapter`` (receives inbound HTTPS POSTs — Slack Events API,
Telegram, Teams, WhatsApp).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# Callback signature: (ctx, adapter) -> awaitable None
MessageCallback = Callable[["MessageContext", "PlatformAdapter"], Awaitable[None]]


class StreamDraftOutcome(Enum):
    """Result of a live-preview draft update — see ``send_stream_draft``.

    Three outcomes, not a bool: the streamer must tell a rendered preview
    (advance the throttle) apart from a transient no-op it should retry next
    chunk without burning the throttle, apart from a permanent stop.
    """

    SHOWN = "shown"
    SKIPPED = "skipped"
    STOPPED = "stopped"


# Where the message came from:
# - "dm"      — 1:1 conversation, reply in-place
# - "channel" — public channel, bot was @mentioned, create a thread to respond
# - "thread"  — ongoing thread conversation, reply in-place
ChannelType = Literal["dm", "channel", "thread"]


@dataclass
class MessageHistoryEntry:
    """A prior platform message included as context for the current turn."""

    username: str
    user_id: Optional[str]
    text: str


@dataclass
class ReferencedConversation:
    """A different thread/channel the incoming message linked or @-referenced,
    fetched by the bot via its own gateway so the model can read it directly
    instead of trying to web-fetch a JS-rendered Discord page.

    ``channel_id`` ties this content back to the raw link/mention in the user's
    message so the renderer can rewrite that link into a readable ``#name``.
    ``messages`` is in chronological order and already bounded to a char budget.
    """

    title: str
    channel_id: str
    messages: tuple[MessageHistoryEntry, ...]


class FileAttachment(BaseModel):
    """A workspace artifact ready to attach to a platform message.

    ``content`` carries the file bytes — the handler only ever produces
    these after the backend has already checked them against the adapter's
    ``max_attachment_bytes``, so adapters can attach directly without
    re-validating size.
    """

    filename: str
    mime_type: str
    content: bytes


class InboundAttachment(BaseModel):
    """A file the user attached to an inbound platform message.

    The adapter downloads the bytes from the platform up-front (bounded by the
    adapter's ``max_attachment_bytes``); the handler then uploads them to the
    user's workspace so AutoPilot can read them during the turn.
    """

    filename: str
    mime_type: str
    content: bytes


class ChannelInfo(BaseModel):
    """A channel the bot can post to, scoped to a server it's connected to.

    Returned by ``list_text_channels`` so the proactive-output path (and the
    copilot tool above it) can resolve a human channel reference like
    ``#announcements`` to a concrete ``id`` and present a picker.
    """

    id: str
    name: str
    server_id: str
    server_name: Optional[str] = None


class PostedRef(BaseModel):
    """Pointer to something the bot just created on the platform.

    ``url`` is a best-effort permalink (Discord ``jump_url``) so callers can
    surface a clickable link in their confirmation; platforms without
    permalinks leave it ``None``.
    """

    id: str
    url: Optional[str] = None


@dataclass
class MessageContext:
    """Everything the core handler needs to know about an incoming message."""

    platform: str
    channel_type: ChannelType
    server_id: Optional[str]
    channel_id: str  # DM channel ID / parent channel ID / thread ID
    message_id: str  # the incoming message itself — used to create threads from it
    user_id: str
    username: str
    text: str  # with bot mentions stripped
    bot_mentioned: bool = False
    thread_history: tuple[MessageHistoryEntry, ...] = ()
    # Users the bot is allowed to @-mention back in this turn — populated
    # from the inbound platform message's mentions (excluding the bot itself).
    # `(display_name, platform_user_id)` pairs. Anyone not in this list won't
    # get pinged even if the LLM produces `@theirname` in its output.
    mentionable_users: tuple[tuple[str, str], ...] = ()
    # Other threads/channels the message linked or @-referenced, fetched by the
    # bot up-front so the model has their content without web-fetching Discord.
    referenced_conversations: tuple[ReferencedConversation, ...] = ()
    # Files the user attached to this message (bytes already downloaded). The
    # handler uploads these to the workspace and passes their IDs to the turn.
    attachments: tuple[InboundAttachment, ...] = ()
    # Attachments the adapter couldn't ingest at all (too large / download
    # failed), as ``(filename, reason)`` pairs — surfaced to the user and the
    # model so neither thinks the file was read.
    skipped_attachments: tuple[tuple[str, str], ...] = ()

    @property
    def is_dm(self) -> bool:
        return self.channel_type == "dm"


class PlatformAdapter(ABC):
    """Outbound contract shared by every platform adapter.

    The core handler only ever holds a ``PlatformAdapter`` — it speaks through
    these methods and never cares whether events arrive over a socket or a
    webhook. Inbound wiring lives on the two subtypes below.
    """

    @property
    @abstractmethod
    def platform_name(self) -> str: ...

    @abstractmethod
    def on_message(self, callback: MessageCallback) -> None: ...

    @abstractmethod
    async def send_message(
        self,
        channel_id: str,
        text: str,
        mentionable_users: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """Send `text` to `channel_id`.

        `mentionable_users` is the allowlist of `(display_name, user_id)` pairs
        the bot may ping in this message. Adapters should resolve `@DisplayName`
        in the text to a real platform mention only for users on this list — a
        defence against the LLM hallucinating mentions or pinging unrelated
        users it learned about elsewhere. Default empty = render mentions as
        plain text.
        """
        ...

    @abstractmethod
    async def send_link(
        self, channel_id: str, text: str, link_label: str, link_url: str
    ) -> None:
        """Send a message with a clickable link presented as a button/CTA.

        Platforms without native button support should fall back to rendering
        the URL inline in the text.
        """
        ...

    @abstractmethod
    async def send_reply(
        self,
        channel_id: str,
        text: str,
        reply_to_message_id: str,
        mentionable_users: tuple[tuple[str, str], ...] = (),
    ) -> None: ...

    @abstractmethod
    async def send_ephemeral(
        self, channel_id: str, user_id: str, text: str
    ) -> None: ...

    async def start_typing(self, channel_id: str) -> None:
        """Show the platform's typing indicator. Default no-op — several
        platforms (Slack bot apps, most webhook platforms) don't expose one."""

    async def stop_typing(self, channel_id: str) -> None:
        """Clear the typing indicator. Default no-op (see ``start_typing``)."""

    @property
    def supports_stream_drafts(self) -> bool:
        """Whether ``send_stream_draft`` can show a live in-progress preview.

        Default False — only platforms with a native draft-streaming API
        (Telegram ``sendMessageDraft``) override this. When False the shared
        streamer never calls ``send_stream_draft``.
        """
        return False

    async def send_stream_draft(
        self, channel_id: str, draft_id: int, text: str
    ) -> StreamDraftOutcome:
        """Show/update an ephemeral preview of the reply being generated.

        Repeated calls with the same nonzero ``draft_id`` update the preview
        in place. The finished reply is still delivered through the normal
        send path, which supersedes the preview — a failed or skipped draft
        never loses content. Returns ``SHOWN`` when the preview rendered,
        ``SKIPPED`` for a transient no-op the caller should retry on the next
        chunk, or ``STOPPED`` (unsupported chat, API error) to stop drafting
        for the turn. Default: unsupported.
        """
        return StreamDraftOutcome.STOPPED

    @abstractmethod
    async def create_thread(
        self, channel_id: str, message_id: str, name: str
    ) -> Optional[str]:
        """Create a thread from a message. Returns the thread ID, or None if
        the platform doesn't support threads or creation failed.
        """
        ...

    async def rename_thread(self, thread_id: str, name: str) -> bool:
        """Rename a platform thread/conversation. Default: unsupported —
        platforms with unnamed/implicit threads (Slack) simply inherit this."""
        return False

    @property
    @abstractmethod
    def max_message_length(self) -> int:
        """Hard platform cap on a single message's content length."""
        ...

    @property
    @abstractmethod
    def chunk_flush_at(self) -> int:
        """Flush the streaming buffer once it reaches this length.

        Should be slightly under max_message_length to leave headroom for
        any trailing content that the splitter might pull into the current
        chunk.
        """
        ...

    @property
    @abstractmethod
    def max_attachment_bytes(self) -> int:
        """Hard platform cap on a single uploaded file's size in bytes."""
        ...

    @property
    @abstractmethod
    def max_thread_name_length(self) -> int:
        """Hard platform cap on a thread/topic name's length.

        The shared thread-naming logic clamps candidate titles to this before
        handing them to ``create_thread``/``rename_thread``. Platforms without
        named threads (Slack) can return any value — the name is unused there.
        """
        ...

    @property
    @abstractmethod
    def typing_refresh_interval(self) -> float:
        """Seconds between typing-indicator refreshes while a turn streams.

        Platform typing indicators auto-expire (Discord ~10s, Telegram ~5s);
        the shared keep-alive loop re-fires at this cadence. Platforms with no
        typing indicator can return any value — ``start_typing`` is a no-op.
        """
        ...

    @abstractmethod
    async def send_file(self, channel_id: str, text: str, file: FileAttachment) -> None:
        """Send a single file as an attachment, with optional accompanying text.

        Callers must ensure ``len(file.content) <= max_attachment_bytes`` —
        the handler enforces that upstream via the workspace fetch path.
        """
        ...

    def localize_markup(self, text: str) -> str:
        """Convert the bot's canonical markup into this platform's dialect.

        The core handler and the model speak one canonical flavour (CommonMark:
        ``**bold**``, ``[label](url)``, ``- bullets``). Each adapter converts
        that to what its platform renders (Discord ≈ CommonMark, so the default
        identity is correct; Slack overrides to mrkdwn). Adapters MUST apply
        this to outbound text in their ``send_*`` methods so no send path
        bypasses it — the default here keeps text unchanged.
        """
        return text

    # -- Proactive output (backend → platform) ----------------------------
    # These power scheduled / autopilot-initiated posts, where the bot speaks
    # without a triggering user message. Authorization (which servers a user
    # may post to) is enforced one layer up; adapters here only translate an
    # already-authorized request into platform API calls.

    @abstractmethod
    def looks_like_channel_id(self, ref: str) -> bool:
        """Whether ``ref`` is one of this platform's channel IDs (vs a name).

        The shared proactive-post resolver uses this to decide whether to treat
        a target reference as a raw channel ID or a channel name to look up.
        Each platform's ID grammar differs (Discord numeric snowflakes, Slack
        ``C0123ABCD``, Telegram signed integers), so the discrimination can't
        live in the platform-agnostic layer.
        """
        ...

    @abstractmethod
    async def list_text_channels(
        self, server_ids: tuple[str, ...]
    ) -> list[ChannelInfo]:
        """List the text channels the bot can post to across ``server_ids``.

        Only channels the bot can actually send in should be returned, so the
        caller's picker never offers a channel the post would fail on.
        """
        ...

    @abstractmethod
    async def get_channel_server_id(self, channel_id: str) -> Optional[str]:
        """Return the server (guild) ID a channel belongs to, or ``None``.

        Used to authorize a post target given a raw channel ID without
        enumerating every channel first.
        """
        ...

    async def open_dm_channel(self, platform_user_id: str) -> Optional[str]:
        """Open (or fetch) the bot's 1:1 DM channel with ``platform_user_id``.

        Returns a channel ID usable with ``post_channel_message``, or ``None``
        when the platform/configuration can't DM this user. Default:
        unsupported — the proactive DM path then reports the platform can't
        deliver DMs rather than attempting a send.
        """
        return None

    @abstractmethod
    async def post_channel_message(
        self, channel_id: str, text: str
    ) -> Optional[PostedRef]:
        """Post a standalone message to ``channel_id``.

        Distinct from ``send_message`` (the streaming-reply path): this
        returns a ``PostedRef`` so proactive callers can report what was
        created. Returns ``None`` if the channel can't be posted to.
        """
        ...

    @abstractmethod
    async def create_channel_thread(
        self, channel_id: str, name: str, text: str
    ) -> Optional[PostedRef]:
        """Create a standalone thread in ``channel_id`` and post ``text`` in it.

        "Standalone" = not anchored to a pre-existing message (unlike
        ``create_thread``). Returns the thread's ``PostedRef``, or ``None`` if
        the platform/channel doesn't support it.
        """
        ...


class SocketAdapter(PlatformAdapter):
    """Adapter that owns a long-lived connection (Discord Gateway, Slack Socket
    Mode). The connection is a per-token singleton held open for the process's
    lifetime, so each socket adapter runs in the ``copilot-bot`` pod and is
    driven by ``start``/``stop``.
    """

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...


class WebhookAdapter(PlatformAdapter):
    """Adapter driven by inbound HTTPS webhooks (Slack Events API, Telegram,
    Teams, WhatsApp). Stateless — its routes mount on the main backend API and
    ride the existing N-replica deployment, so no dedicated pod is needed.
    """

    @abstractmethod
    def register_routes(self, app: FastAPI) -> None:
        """Mount this adapter's inbound webhook route(s) onto ``app``.

        The adapter owns its own request signature verification and must ACK
        within the platform's timeout, scheduling the real work off-request.
        """
        ...


async def read_verified_webhook_body(
    request: Request, verify: Callable[[Request, bytes], bool]
) -> bytes | None:
    """Read an inbound webhook body and verify its platform signature.

    Returns the raw body when the signature checks out, ``None`` otherwise —
    the route should then return :func:`unauthorized_webhook_response`. Shared
    so every webhook adapter verifies BEFORE parsing, against the same raw
    bytes the platform signed.
    """
    raw = await request.body()
    if not verify(request, raw):
        return None
    return raw


def unauthorized_webhook_response() -> PlainTextResponse:
    """The uniform 401 for a webhook that failed signature verification."""
    return PlainTextResponse("invalid signature", status_code=401)
