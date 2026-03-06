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
    parentUuid: str | None
    message: dict[str, Any]


class TranscriptBuilder:
    """Build complete JSONL transcript from SDK messages.

    This builder maintains the FULL conversation state, not incremental changes.
    The output is always the complete active context.
    """

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []
        self._last_uuid: str | None = None
        self._pending_tool_results: list[dict[str, Any]] = []

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

            # Load all non-strippable entries (user/assistant/system/etc.)
            # Skip only STRIPPABLE_TYPES to match strip_progress_entries() behavior
            entry_type = data.get("type", "")
            if entry_type in STRIPPABLE_TYPES:
                continue

            entry = TranscriptEntry(
                type=data["type"],
                uuid=data.get("uuid") or str(uuid4()),
                parentUuid=data.get("parentUuid"),
                message=data.get("message", {}),
            )
            self._entries.append(entry)
            self._last_uuid = entry.uuid

        logger.info(
            "%s Loaded %d entries from previous transcript (last_uuid=%s)",
            log_prefix,
            len(self._entries),
            self._last_uuid[:12] if self._last_uuid else None,
        )

    def add_user_message(
        self, content: str | list[dict], uuid: str | None = None
    ) -> None:
        """Add user message to the complete context."""
        msg_uuid = uuid or str(uuid4())

        self._entries.append(
            TranscriptEntry(
                type="user",
                uuid=msg_uuid,
                parentUuid=self._last_uuid,
                message={"role": "user", "content": content},
            )
        )
        self._last_uuid = msg_uuid

    def add_tool_result(self, tool_use_id: str, content: str) -> None:
        """Accumulate a tool result. Flushed automatically before the next assistant message."""
        self._pending_tool_results.append(
            {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
        )

    def flush_pending_tool_results(self) -> None:
        """Flush any pending tool results as a single user message."""
        if self._pending_tool_results:
            self.add_user_message(content=list(self._pending_tool_results))
            self._pending_tool_results.clear()

    def add_assistant_message(
        self, content_blocks: list[dict], model: str = ""
    ) -> None:
        """Add assistant message to the complete context.

        Automatically flushes pending tool results first to maintain
        correct transcript ordering (tool_results → assistant).
        """
        self.flush_pending_tool_results()
        msg_uuid = str(uuid4())

        self._entries.append(
            TranscriptEntry(
                type="assistant",
                uuid=msg_uuid,
                parentUuid=self._last_uuid,
                message={
                    "role": "assistant",
                    "model": model,
                    "content": content_blocks,
                },
            )
        )
        self._last_uuid = msg_uuid

    def _merge_consecutive_assistant_entries(self) -> list[TranscriptEntry]:
        """Merge consecutive assistant entries into single messages.

        The Claude API rejects consecutive messages of the same role.
        The SDK streams thinking, text, and tool_use as separate
        AssistantMessages — this merges them into one entry.
        """
        merged: list[TranscriptEntry] = []
        for entry in self._entries:
            if entry.type == "assistant" and merged and merged[-1].type == "assistant":
                merged[-1].message["content"].extend(entry.message["content"])
            else:
                merged.append(entry)
        return merged

    def to_jsonl(self) -> str:
        """Export complete context as JSONL.

        Returns the FULL conversation state (all entries), not incremental.
        This output REPLACES any previous transcript.
        """
        self.flush_pending_tool_results()
        entries = self._merge_consecutive_assistant_entries()
        if not entries:
            return ""

        lines = [entry.model_dump_json(exclude_none=True) for entry in entries]
        return "\n".join(lines) + "\n"

    @property
    def entry_count(self) -> int:
        """Total number of entries in the complete context."""
        return len(self._entries)

    @property
    def is_empty(self) -> bool:
        """Whether this builder has any entries."""
        return len(self._entries) == 0
