"""Encoding of the opaque target id the handler passes back to sends.

A target is a chat, optionally pinned to a forum topic: ``chat_id`` or
``chat_id|thread_id``. Shared by the adapter (context building, sends) and
the command handlers (session-scoped commands like /new).
"""

from typing import Optional


def encode_target(chat_id: str, thread_id: Optional[int] = None) -> str:
    return f"{chat_id}|{thread_id}" if thread_id is not None else chat_id


def decode_target(target_id: str) -> tuple[str, Optional[int]]:
    chat_id, sep, thread = target_id.partition("|")
    if sep and thread.isdigit():
        return chat_id, int(thread)
    return chat_id, None
