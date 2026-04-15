"""Build complete JSONL transcript from SDK messages.

The transcript represents the FULL active context at any point in time.
Each upload REPLACES the previous transcript atomically.

Flow:
  Turn 1: Upload [msg1, msg2]
  Turn 2: Download [msg1, msg2] → Upload [msg1, msg2, msg3, msg4] (REPLACE)
  Turn 3: Download [msg1, msg2, msg3, msg4] → Upload [all messages] (REPLACE)

The transcript is never incremental - always the complete atomic state.
"""

import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from backend.util import json

from .transcript import STRIPPABLE_TYPES

logger = logging.getLogger(__name__)


class TranscriptEntry(BaseModel):
    """Single transcript entry (user or assistant turn)."""

    type: str
    uuid: str
    parentUuid: str = ""
    isCompactSummary: bool | None = None
    message: dict[str, Any]


class TranscriptBuilder:
    """Build complete JSONL transcript from SDK messages.

    This builder maintains the FULL conversation state, not incremental changes.
    The output is always the complete active context.
    """

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []
        self._last_uuid: str | None = None

    def _last_is_assistant(self) -> bool:
        return bool(self._entries) and self._entries[-1].type == "assistant"

    def _last_message_id(self) -> str:
        """Return the message.id of the last entry, or '' if none."""
        if self._entries:
            return self._entries[-1].message.get("id", "")
        return ""

    @staticmethod
    def _parse_entry(data: dict) -> TranscriptEntry | None:
        """Parse a single transcript entry, filtering strippable types.

        Returns ``None`` for entries that should be skipped (strippable types
        that are not compaction summaries).
        """
        entry_type = data.get("type", "")
        if entry_type in STRIPPABLE_TYPES and not data.get("isCompactSummary"):
            return None
        return TranscriptEntry(
            type=entry_type,
            uuid=data.get("uuid") or str(uuid4()),
            parentUuid=data.get("parentUuid") or "",
            isCompactSummary=data.get("isCompactSummary"),
            message=data.get("message", {}),
        )

    def load_previous(self, content: str, log_prefix: str = "[Transcript]") -> None:
        """Load complete previous transcript.

        This loads the FULL previous context. As new messages come in,
        we append to this state. The final output is the complete context
        (previous + new), not just the delta.
        """
        if not content or not content.strip():
            return

        lines = content.strip().split("\n")
        for line_num, line in enumerate(lines, 1):
            if not line.strip():
                continue

            data = json.loads(line, fallback=None)
            if data is None:
                logger.warning(
                    "%s Failed to parse transcript line %d/%d",
                    log_prefix,
                    line_num,
                    len(lines),
                )
                continue

            entry = self._parse_entry(data)
            if entry is None:
                continue
            self._entries.append(entry)
            self._last_uuid = entry.uuid

        logger.info(
            "%s Loaded %d entries from previous transcript (last_uuid=%s)",
            log_prefix,
            len(self._entries),
            self._last_uuid[:12] if self._last_uuid else None,
        )

    def append_user(self, content: str | list[dict], uuid: str | None = None) -> None:
        """Append a user entry."""
        msg_uuid = uuid or str(uuid4())

        self._entries.append(
            TranscriptEntry(
                type="user",
                uuid=msg_uuid,
                parentUuid=self._last_uuid or "",
                message={"role": "user", "content": content},
            )
        )
        self._last_uuid = msg_uuid

    def append_tool_result(self, tool_use_id: str, content: str) -> None:
        """Append a tool result as a user entry (one per tool call)."""
        self.append_user(
            content=[
                {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
            ]
        )

    def append_assistant(
        self,
        content_blocks: list[dict],
        model: str = "",
        stop_reason: str | None = None,
    ) -> None:
        """Append an assistant entry.

        Consecutive assistant entries automatically share the same message ID
        so the CLI can merge them (thinking → text → tool_use) into a single
        API message on ``--resume``.  A new ID is assigned whenever an
        assistant entry follows a non-assistant entry (user message or tool
        result), because that marks the start of a new API response.
        """
        message_id = (
            self._last_message_id()
            if self._last_is_assistant()
            else f"msg_sdk_{uuid4().hex[:24]}"
        )

        msg_uuid = str(uuid4())

        self._entries.append(
            TranscriptEntry(
                type="assistant",
                uuid=msg_uuid,
                parentUuid=self._last_uuid or "",
                message={
                    "role": "assistant",
                    "model": model,
                    "id": message_id,
                    "type": "message",
                    "content": content_blocks,
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                },
            )
        )
        self._last_uuid = msg_uuid

    def replace_entries(
        self, compacted_entries: list[dict], log_prefix: str = "[Transcript]"
    ) -> None:
        """Replace all entries with compacted entries from the CLI session file.

        Called after mid-stream compaction so TranscriptBuilder mirrors the
        CLI's active context (compaction summary + post-compaction entries).

        Builds the new list first and validates it's non-empty before swapping,
        so corrupt input cannot wipe the conversation history.
        """
        new_entries: list[TranscriptEntry] = []
        for data in compacted_entries:
            entry = self._parse_entry(data)
            if entry is not None:
                new_entries.append(entry)

        if not new_entries:
            logger.warning(
                "%s replace_entries produced 0 entries from %d inputs, keeping old (%d entries)",
                log_prefix,
                len(compacted_entries),
                len(self._entries),
            )
            return

        old_count = len(self._entries)
        self._entries = new_entries
        self._last_uuid = new_entries[-1].uuid

        logger.info(
            "%s TranscriptBuilder compacted: %d entries -> %d entries",
            log_prefix,
            old_count,
            len(self._entries),
        )

    def to_jsonl(self) -> str:
        """Export complete context as JSONL.

        Consecutive assistant entries are kept separate to match the
        native CLI format — the SDK merges them internally on resume.

        Returns the FULL conversation state (all entries), not incremental.
        This output REPLACES any previous transcript.
        """
        if not self._entries:
            return ""

        lines = [entry.model_dump_json(exclude_none=True) for entry in self._entries]
        return "\n".join(lines) + "\n"

    def snapshot(self) -> tuple[list[TranscriptEntry], str | None]:
        """Return a shallow snapshot of the current builder state.

        Use with :meth:`restore` to roll back transcript mutations from a
        failed stream attempt without accessing private attributes directly.

        Returns a ``(entries_copy, last_uuid)`` tuple.  ``entries_copy`` is a
        new list (shallow copy) so caller mutations don't affect the live state.
        """
        return list(self._entries), self._last_uuid

    def restore(self, snap: tuple[list[TranscriptEntry], str | None]) -> None:
        """Restore builder state from a :meth:`snapshot`.

        Replaces ``_entries`` and ``_last_uuid`` atomically so the builder
        matches the state at the time the snapshot was taken.
        """
        self._entries, self._last_uuid = snap

    @property
    def entry_count(self) -> int:
        """Total number of entries in the complete context."""
        return len(self._entries)

    @property
    def is_empty(self) -> bool:
        """Whether this builder has any entries."""
        return len(self._entries) == 0

    @property
    def last_entry_type(self) -> str | None:
        """Type of the last entry, or None if empty."""
        return self._entries[-1].type if self._entries else None
