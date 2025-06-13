"""Embedded Solenne companion package."""

from .config import CFG, Config
from .ai import get_response, store_async, async_messages, handle_user

__all__ = [
    "CFG",
    "Config",
    "get_response",
    "store_async",
    "async_messages",
    "handle_user",
]
