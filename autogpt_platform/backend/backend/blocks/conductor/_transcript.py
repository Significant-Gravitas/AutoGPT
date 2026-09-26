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

from ._api import PAGE_SIZE, ConductorClient
from ._paging import (
    Remaining,
    bounded,
    expired,
    fetch_after,
    fetch_before,
    fetch_tail_at,
)

# Rows kept for one reply. A longer turn keeps its newest rows (the answer is
# at the end) and reports `truncated`.
MAX_TURN_MESSAGES = 1000
# How far back the first poll looks for the prompt row. A just-sent prompt is
# among the newest rows; if it is not there yet it has not been recorded.
PROMPT_SCAN_MESSAGES = 300
# When neither the prompt row nor a row of its turn is among the newest
# PROMPT_SCAN_MESSAGES, this many older rows are read before giving up on the
# history and waiting for the turn to arrive.
PROMPT_SEARCH_MESSAGES = MAX_TURN_MESSAGES

USER_TYPES = frozenset({"usermessage", "user", "human", "prompt"})
NON_AGENT_TYPES = USER_TYPES | {"system"}
# Harness events that only mark a turn as launched. Claude Code's `system`
# (init) and Conductor's `command_lifecycle` rows, and Codex's thread/turn
# started events, say nothing about whether the agent produced anything.
STARTUP_RAW_TYPES = frozenset({"system", "command_lifecycle"})
STARTUP_CODEX_EVENTS = frozenset({"thread.started", "turn.started"})


async def wait_for_reply(
    client: ConductorClient,
    session_id: str,
    prompt_message_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    """Wait until the agent has answered the prompt, then return its turn.

    Each poll reads the transcript rows that arrived since the previous poll
    and then the session status, so the status is never older than the rows
    it is judged against. The wait ends when the session reports `error`, or
    `idle` after the prompt's turn has progressed past its startup events
    (an idle session whose prompt is still queued, or that has only launched
    the agent, is not a finished one); the transcript is then read once more
    for rows written just before the status changed. Sleeps and requests
    never outlive `timeout_seconds`: nothing is started once it has elapsed,
    and a reply that completes after it is reported as timed out.
    """
    deadline = time.monotonic() + timeout_seconds

    def remaining() -> float:
        return deadline - time.monotonic()

    turn = _TurnCollector(prompt_message_id)
    status: dict[str, Any] = {}
    timed_out = False
    while True:
        await asyncio.sleep(min(poll_interval_seconds, max(0.0, remaining())))
        if expired(remaining):
            timed_out = True
            break
        try:
            await turn.refresh(client, session_id, remaining)
            status = await bounded(client.session_status(session_id), remaining)
        except TimeoutError:
            timed_out = True
            break
        if expired(remaining):
            timed_out = True
            break
        state = str(status.get("status") or "")
        if state == "error" or (state == "idle" and turn.progressed):
            try:
                await turn.refresh(client, session_id, remaining)
            except TimeoutError:
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
    """Accumulates the rows of one prompt's turn across polls.

    The turn is resolved by its prompt row (`content.id` or row id equal to
    the receipt) or, when that row is older than the rows read, by agent rows
    tagged with the receipt as their `turnId`/`userMessageId`; in the latter
    case the omitted history is reported through `truncated`.
    """

    def __init__(self, receipt_id: str):
        self.receipt_id = receipt_id
        self.turn_id = ""
        self.resolved = False
        self.progressed = False
        self.truncated = False
        self.cursor = ""
        self.rows: deque[dict[str, Any]] = deque(maxlen=MAX_TURN_MESSAGES)

    async def refresh(
        self, client: ConductorClient, session_id: str, remaining: Remaining
    ) -> None:
        """Consume every row that arrived since the last refresh, one page at
        a time so rows already read survive a request that hits the deadline."""
        if not self.cursor:
            await self._locate(client, session_id, remaining)
            return
        while True:
            rows, has_more = await fetch_after(
                client, session_id, self.cursor, PAGE_SIZE, remaining
            )
            self._consume_all(rows)
            if not has_more or expired(remaining):
                return

    async def _locate(
        self, client: ConductorClient, session_id: str, remaining: Remaining
    ) -> None:
        """First read: the newest rows, then bounded older pages until one
        belongs to the turn. Rows are consumed oldest first only once the
        search stops, so unresolved history is never skipped past."""
        rows, start = await fetch_tail_at(
            client, session_id, PROMPT_SCAN_MESSAGES, remaining
        )
        found = any(self._belongs(row) for row in rows)
        budget = PROMPT_SEARCH_MESSAGES
        while not found and start > 0 and budget > 0 and not expired(remaining):
            page, start = await fetch_before(
                client, session_id, start, min(PAGE_SIZE, budget), remaining
            )
            budget -= max(len(page), 1)
            found = any(self._belongs(row) for row in page)
            rows = page + rows
        self._consume_all(rows)

    def _consume_all(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._consume(row)
        if rows:
            self.cursor = str(rows[-1].get("id") or self.cursor)

    def _consume(self, row: dict[str, Any]) -> None:
        if not self.resolved:
            if self._is_prompt(row):
                self.turn_id = _turn_of(row) or self.receipt_id
            elif self.receipt_id in _tags(row):
                # The prompt row is older than anything read: the turn's
                # start is not in `rows`.
                self.turn_id = self.receipt_id
                self.truncated = True
            else:
                return
            self.resolved = True
        elif not self._same_turn(row):
            return
        if is_agent_message(row) and not _is_startup(row):
            self.progressed = True
        self._keep(row)

    def _belongs(self, row: dict[str, Any]) -> bool:
        return self._is_prompt(row) or self.receipt_id in _tags(row)

    def _is_prompt(self, row: dict[str, Any]) -> bool:
        return (
            _content(row).get("id") == self.receipt_id
            or row.get("id") == self.receipt_id
        )

    def _same_turn(self, row: dict[str, Any]) -> bool:
        tags = _tags(row)
        return not tags or self.turn_id in tags or self.receipt_id in tags

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


def _tags(row: dict[str, Any]) -> frozenset[str]:
    """The turn identifiers a row carries (`turnId`, `userMessageId`)."""
    content = _content(row)
    return frozenset(
        str(tag) for tag in (content.get("turnId"), content.get("userMessageId")) if tag
    )


def _is_startup(row: dict[str, Any]) -> bool:
    raw = _raw_payload(row)
    if raw is None:
        return False
    if raw.get("type") in STARTUP_RAW_TYPES:
        return True
    event = raw.get("event")
    return isinstance(event, dict) and event.get("type") in STARTUP_CODEX_EVENTS


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
