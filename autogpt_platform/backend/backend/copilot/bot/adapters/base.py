"""Abstract base for platform adapters.

Each chat platform (Discord, Telegram, Slack, etc.) implements this interface.
The core bot logic in handler.py is platform-agnostic — it only speaks through
these methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable, Literal, Optional

# Callback signature: (ctx, adapter) -> awaitable None
MessageCallback = Callable[["MessageContext", "PlatformAdapter"], Awaitable[None]]

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

    @property
    def is_dm(self) -> bool:
        return self.channel_type == "dm"


class PlatformAdapter(ABC):
    """Interface that each chat platform must implement."""

    @property
    @abstractmethod
    def platform_name(self) -> str: ...

    @abstractmethod
    def on_message(self, callback: MessageCallback) -> None: ...

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

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

    @abstractmethod
    async def start_typing(self, channel_id: str) -> None: ...

    @abstractmethod
    async def stop_typing(self, channel_id: str) -> None: ...

    @abstractmethod
    async def create_thread(
        self, channel_id: str, message_id: str, name: str
    ) -> Optional[str]:
        """Create a thread from a message. Returns the thread ID, or None if
        the platform doesn't support threads or creation failed.
        """
        ...

    @abstractmethod
    async def rename_thread(self, thread_id: str, name: str) -> bool:
        """Rename a platform thread/conversation when supported."""
        ...

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
