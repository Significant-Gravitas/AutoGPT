"""Parse Conductor transcript rows and wait for an agent's reply.

Transcript rows are `{id, sessionIndex, type, content, receivedAt}`. Live
sessions use two row types: `userMessage`, whose `content.id` is the receipt
id returned when the prompt was sent, and `agent`, whose `content.rawPayload`
is the harness's own event (Claude Code stream-json or Codex app-server
events) tagged with the `turnId`/`userMessageId` it belongs to. Row ids and
receipt ids are different namespaces: only row ids work as `after` cursors.
"""

import asyncio
import time
from collections import deque
from typing import Any

from ._api import ConductorClient
from ._paging import Remaining, bounded, expired, fetch_after, fetch_tail

# Rows kept for one reply. A longer turn keeps its newest rows (the answer is
# at the end) and reports `truncated`.
MAX_TURN_MESSAGES = 1000
# How far back the first poll looks for the prompt row. A just-sent prompt is
# among the newest rows; if it is not there yet it has not been recorded.
PROMPT_SCAN_MESSAGES = 300

USER_TYPES = frozenset({"usermessage", "user", "human", "prompt"})
NON_AGENT_TYPES = USER_TYPES | {"system"}


async def wait_for_reply(
    client: ConductorClient,
    session_id: str,
    prompt_message_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    """Wait until the agent has answered the prompt, then return its turn.

    Each poll reads the session status and the transcript rows that arrived
    since the previous poll. The wait ends when the session reports `error`,
    or `idle` after at least one agent row of the prompt's turn was seen, so
    an idle session that has not started the prompt yet is not mistaken for a
    finished one. Sleeps and requests never outlive `timeout_seconds`.
    """
    deadline = time.monotonic() + timeout_seconds

    def remaining() -> float:
        return deadline - time.monotonic()

    turn = _TurnCollector(prompt_message_id)
    status: dict[str, Any] = {}
    timed_out = False
    while True:
        await asyncio.sleep(min(poll_interval_seconds, max(0.0, remaining())))
        try:
            status = await bounded(client.session_status(session_id), remaining)
            await turn.refresh(client, session_id, remaining)
        except TimeoutError:
            timed_out = True
            break
        state = str(status.get("status") or "")
        if state == "error" or (state == "idle" and turn.started):
            break
        if expired(remaining):
            timed_out = True
            break
    messages = list(turn.rows)
    return {
        "session_status": str(status.get("status") or ""),
        "error_message": str(
            status.get("errorMessage") or status.get("lastError") or ""
        ),
        "messages": messages,
        "reply": reply_text(messages),
        "timed_out": timed_out,
        "truncated": turn.truncated,
    }


def reply_text(messages: list[dict[str, Any]]) -> str:
    """All visible agent text of a transcript slice, oldest first."""
    texts = [text for message in messages if (text := message_text(message))]
    return "\n\n".join(texts) if texts else _result_text(messages)


def latest_reply(messages: list[dict[str, Any]]) -> str:
    """Text of the newest text-bearing agent message in the slice."""
    for message in reversed(messages):
        if text := message_text(message):
            return text
    return _result_text(messages)


def message_text(message: dict[str, Any]) -> str:
    """Visible assistant text of one row; empty for prompts, tool calls and
    results, reasoning and lifecycle events, or shapes we do not recognise."""
    if not is_agent_message(message):
        return ""
    raw = _raw_payload(message)
    if raw is not None:
        return _raw_text(raw)
    return _plain_text(message.get("content"))


def is_agent_message(message: dict[str, Any]) -> bool:
    return str(message.get("type") or "").lower() not in NON_AGENT_TYPES


class _TurnCollector:
    """Accumulates the rows of one prompt's turn across polls."""

    def __init__(self, receipt_id: str):
        self.receipt_id = receipt_id
        self.turn_id = ""
        self.found_prompt = False
        self.started = False
        self.truncated = False
        self.cursor = ""
        self.rows: deque[dict[str, Any]] = deque(maxlen=MAX_TURN_MESSAGES)

    async def refresh(
        self, client: ConductorClient, session_id: str, remaining: Remaining
    ) -> None:
        """Consume every row that arrived since the last refresh."""
        while True:
            if self.cursor:
                rows, has_more = await fetch_after(
                    client, session_id, self.cursor, MAX_TURN_MESSAGES, remaining
                )
            else:
                rows, _ = await fetch_tail(
                    client, session_id, PROMPT_SCAN_MESSAGES, remaining
                )
                has_more = False
            for row in rows:
                self._consume(row)
            if rows:
                self.cursor = str(rows[-1].get("id") or self.cursor)
            if not has_more or expired(remaining):
                return

    def _consume(self, row: dict[str, Any]) -> None:
        if not self.found_prompt:
            if not self._is_prompt(row):
                return
            self.found_prompt = True
            self.turn_id = _turn_of(row) or self.receipt_id
            self._keep(row)
            return
        if not _same_turn(row, self.turn_id):
            return
        if is_agent_message(row):
            self.started = True
        self._keep(row)

    def _is_prompt(self, row: dict[str, Any]) -> bool:
        return (
            _content(row).get("id") == self.receipt_id
            or row.get("id") == self.receipt_id
        )

    def _keep(self, row: dict[str, Any]) -> None:
        if len(self.rows) == self.rows.maxlen:
            self.truncated = True
        self.rows.append(row)


def _content(row: dict[str, Any]) -> dict[str, Any]:
    content = row.get("content")
    return content if isinstance(content, dict) else {}


def _turn_of(row: dict[str, Any]) -> str:
    content = _content(row)
    return str(content.get("turnId") or content.get("userMessageId") or "")


def _same_turn(row: dict[str, Any], turn_id: str) -> bool:
    turn = _turn_of(row)
    return not turn or turn == turn_id


def _raw_payload(message: dict[str, Any]) -> dict[str, Any] | None:
    raw = _content(message).get("rawPayload")
    return raw if isinstance(raw, dict) else None


def _raw_text(raw: dict[str, Any]) -> str:
    """Assistant text of a Claude `assistant` event or a completed Codex
    `agentMessage` item. Every other event carries no visible text."""
    if raw.get("type") == "assistant":
        message = raw.get("message")
        return _text_parts(message.get("content")) if isinstance(message, dict) else ""
    event = raw.get("event")
    if not isinstance(event, dict) or event.get("type") != "item.completed":
        return ""
    item = event.get("item")
    if isinstance(item, dict) and item.get("type") == "agentMessage":
        return str(item.get("text") or "")
    return ""


def _text_parts(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts = [
        str(part["text"])
        for part in content
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
    ]
    return "\n".join(texts)


def _plain_text(content: Any) -> str:
    """Text of the simple shapes: a string, `{text}` or a list of text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        parts = [_plain_text(part) for part in content]
        return "\n".join(part for part in parts if part)
    return ""


def _result_text(messages: list[dict[str, Any]]) -> str:
    """Claude's final `result` event repeats the answer; use it only when no
    assistant text was seen (for example a turn that ended without one)."""
    for message in reversed(messages):
        raw = _raw_payload(message)
        if raw is None or raw.get("type") != "result" or raw.get("is_error"):
            continue
        result = raw.get("result")
        if isinstance(result, str) and result:
            return result
    return ""
