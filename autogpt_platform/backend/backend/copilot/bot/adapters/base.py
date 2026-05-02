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
    async def send_message(self, channel_id: str, text: str) -> None: ...

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
        self, channel_id: str, text: str, reply_to_message_id: str
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
