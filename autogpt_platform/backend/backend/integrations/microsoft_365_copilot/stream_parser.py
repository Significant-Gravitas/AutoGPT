"""Turning Microsoft Graph's Copilot stream into deltas.

Every other chat provider in this codebase streams *deltas*: each event
carries the new text and the client appends it. Microsoft's
``chatOverStream`` streams **cumulative snapshots** -- each event carries a
whole ``copilotConversation`` with the assistant's answer as it stands so
far. Appending those the way you would append a delta produces the answer
repeated once per event, growing quadratically.

So this parser exists to compute what the rest of the pipeline expects: it
remembers what it has already emitted for the assistant's message and yields
only the new suffix.

Three things make that harder than a string comparison:

- A snapshot can arrive with the conversation but no assistant message yet,
  or with an empty one. Those are not "the model said nothing"; they are
  the stream warming up, and emitting for them produces spurious empty
  chunks.
- A conversation carries more than one message. Locking onto the assistant
  response by id, once, is what keeps a later user or system row from being
  mistaken for a continuation of the answer.
- A snapshot can *shrink* or be replaced -- a filtered or regenerated
  answer. Emitting a "suffix" computed against a longer previous string
  would emit nothing forever afterwards, so a snapshot that is not an
  extension of what came before resets rather than diffs.

Byte-level chunk splitting is handled by the caller: SSE frames are
assembled before they reach here, so this sees whole JSON objects.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_ASSISTANT_ROLES = frozenset({"assistant", "bot", "copilot"})


class CopilotStreamParser:
    """Stateful across one turn. Not reusable for a second."""

    def __init__(self) -> None:
        # The assistant message this turn is about. Locked on first sight so
        # a later row in the same conversation cannot be mistaken for a
        # continuation of the answer.
        self._message_id: str | None = None
        self._emitted = ""

    def feed(self, event_data: str) -> str:
        """One SSE ``data:`` payload in, the new text out.

        Returns an empty string for anything that adds nothing -- a warming
        -up snapshot, a duplicate, a heartbeat. The caller can emit
        unconditionally without producing empty chunks.
        """
        payload = self._parse(event_data)
        if payload is None:
            return ""

        message = self._assistant_message(payload)
        if message is None:
            return ""

        text = _text_of(message)
        if not text:
            return ""

        if not text.startswith(self._emitted):
            # Not an extension of what we have shown. A regenerated or
            # filtered answer replaces rather than continues, and diffing
            # against the old one would emit nothing from here on.
            logger.debug(
                "[m365] snapshot is not an extension of the emitted text; resetting"
            )
            self._emitted = text
            return text

        delta = text[len(self._emitted) :]
        self._emitted = text
        return delta

    @property
    def text(self) -> str:
        """Everything emitted so far, for persisting the finished message."""
        return self._emitted

    def _parse(self, event_data: str) -> dict[str, Any] | None:
        raw = event_data.strip()
        # SSE keep-alives and the terminal marker are not JSON and are not
        # errors -- treating them as malformed would log noise on every
        # healthy stream.
        if not raw or raw == "[DONE]":
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("[m365] discarding a stream event that is not JSON")
            return None
        return payload if isinstance(payload, dict) else None

    def _assistant_message(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        conversation = payload.get("copilotConversation")
        if not isinstance(conversation, dict):
            # Graph has also been observed sending the conversation at the
            # top level. Reading only the wrapped shape would drop the whole
            # answer, which is a silent failure rather than a loud one.
            conversation = payload

        messages = conversation.get("messages")
        if not isinstance(messages, list):
            return None

        if self._message_id is not None:
            return _find_by_id(messages, self._message_id)

        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).lower() not in _ASSISTANT_ROLES:
                continue
            # Only lock on once there is something to show. Locking onto an
            # empty placeholder is harmless, but waiting means the id we
            # keep is the one that actually carried the answer.
            if not _text_of(message):
                continue
            self._message_id = str(message.get("id") or "")
            return message
        return None


def _find_by_id(messages: list[Any], message_id: str) -> dict[str, Any] | None:
    for message in messages:
        if isinstance(message, dict) and str(message.get("id") or "") == message_id:
            return message
    return None


def _text_of(message: dict[str, Any]) -> str:
    """The assistant text, across the shapes Graph uses for it."""
    body = message.get("text")
    if isinstance(body, str):
        return body
    # ``body: {content: ...}`` is the shape Graph uses elsewhere for message
    # content, and the Copilot preview has not been stable about which it
    # sends.
    if isinstance(body, dict):
        content = body.get("content")
        if isinstance(content, str):
            return content
    nested = message.get("body")
    if isinstance(nested, dict):
        content = nested.get("content")
        if isinstance(content, str):
            return content
    if isinstance(nested, str):
        return nested
    return ""
