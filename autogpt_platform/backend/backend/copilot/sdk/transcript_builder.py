"""Build complete JSONL transcript from SDK messages.

The transcript represents the FULL active context at any point in time.
Each upload REPLACES the previous transcript atomically.

Flow:
  Turn 1: Upload [msg1, msg2]
  Turn 2: Download [msg1, msg2] → Upload [msg1, msg2, msg3, msg4] (REPLACE)
  Turn 3: Download [msg1, msg2, msg3, msg4] → Upload [all messages] (REPLACE)

The transcript is never incremental - always the complete atomic state.
"""

import json
import logging
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

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

    def load_previous(self, content: str) -> None:
        """Load complete previous transcript.

        This loads the FULL previous context. As new messages come in,
        we append to this state. The final output is the complete context
        (previous + new), not just the delta.
        """
        if not content or not content.strip():
            return

        for line in content.strip().split("\n"):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Failed to parse transcript line: %s", line[:100])
                continue

            # Only load conversation messages (user/assistant)
            # Skip metadata entries
            if data.get("type") not in ("user", "assistant"):
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
            "Loaded %d entries from previous transcript (last_uuid=%s)",
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

    def add_assistant_message(
        self, content_blocks: list[dict], model: str = ""
    ) -> None:
        """Add assistant message to the complete context."""
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

    def to_jsonl(self) -> str:
        """Export complete context as JSONL.

        Returns the FULL conversation state (all entries), not incremental.
        This output REPLACES any previous transcript.
        """
        if not self._entries:
            return ""

        lines = [entry.model_dump_json(exclude_none=True) for entry in self._entries]
        return "\n".join(lines) + "\n"

    @property
    def entry_count(self) -> int:
        """Total number of entries in the complete context."""
        return len(self._entries)

    @property
    def is_empty(self) -> bool:
        """Whether this builder has any entries."""
        return len(self._entries) == 0
