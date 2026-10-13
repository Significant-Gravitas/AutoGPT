"""What the model was asked and what it answered, for a journey to inspect.

``GET /__scripted__/journal?nonce=<nonce>`` returns one run's entries. A
journey reads it to wait for a reply to finish streaming, or to print what a
tool really returned when a check failed.
"""

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from .planner import Conversation, Plan
from .scripts import find_marker

# Enough for a CI run's journeys; the oldest entries go first.
MAX_ENTRIES = 1000
# A tool result can be a whole agent graph; the start says what happened.
MAX_RESULT_CHARS = 2000


@dataclass(frozen=True)
class Request:
    dialect: str
    path: str


class Journal:
    def __init__(self) -> None:
        self._entries: deque[dict[str, Any]] = deque(maxlen=MAX_ENTRIES)
        self._lock = threading.Lock()
        self._next_id = 1

    def record_plan(
        self, request: Request, conversation: Conversation, planned: Plan
    ) -> dict[str, Any]:
        return self._add(
            request,
            conversation,
            kind=planned.kind,
            turn=None if planned.turn_index is None else planned.turn_index + 1,
            turns=planned.turn_count,
            text=planned.reply.text,
            tool_calls=[name for name, _ in planned.reply.tool_calls],
            usage=asdict(planned.reply.usage),
            cost_usd=planned.reply.cost_usd,
            checks=[
                {"what": c.what, "ok": c.ok, "detail": c.detail} for c in planned.checks
            ],
            completed=None,
        )

    def record_error(
        self, request: Request, conversation: Conversation, reason: str
    ) -> dict[str, Any]:
        return self._add(request, conversation, kind="error", error=reason)

    def record_unexpected(self, method: str, path: str) -> dict[str, Any]:
        with self._lock:
            entry = {"id": self._take_id(), "at": time.time(), "kind": "unexpected"}
            entry.update(method=method, path=path)
            self._entries.append(entry)
            return entry

    def finish(self, entry: dict[str, Any], completed: bool) -> None:
        """Mark whether the whole reply reached the platform."""
        with self._lock:
            entry["completed"] = completed
            entry["finished_at"] = time.time()

    def find(
        self, script: str | None = None, nonce: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(entry)
                for entry in self._entries
                if (script is None or entry.get("script") == script)
                and (nonce is None or entry.get("nonce") == nonce)
            ]

    def _add(
        self, request: Request, conversation: Conversation, **fields: Any
    ) -> dict[str, Any]:
        marker = find_marker(conversation.user_text)
        with self._lock:
            entry: dict[str, Any] = {
                "id": self._take_id(),
                "at": time.time(),
                "dialect": request.dialect,
                "path": request.path,
                "model": conversation.model,
                "script": marker.script if marker else None,
                "nonce": marker.nonce if marker else None,
                "tool_results": [
                    {
                        "tool": r.tool,
                        "is_error": r.is_error,
                        "content": r.content[:MAX_RESULT_CHARS],
                    }
                    for r in conversation.tool_results
                ],
            }
            entry.update(fields)
            self._entries.append(entry)
            return entry

    def _take_id(self) -> int:
        entry_id = self._next_id
        self._next_id += 1
        return entry_id
