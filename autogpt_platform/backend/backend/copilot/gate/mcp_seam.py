"""Gate for MCP tools that are not in ``TOOL_REGISTRY``.

``create_copilot_mcp_server`` registers the file handlers from
``sdk/e2b_file_tools.py`` directly, so they are not ``BaseTool`` subclasses and
never reach the seam in ``BaseTool.execute``. Without this they would be the
one write path in SDK mode with no gate in front of it.

Returns an MCP-shaped error rather than a ``StreamToolOutputAvailable``,
because the caller here is the raw handler wrapper, not the tool layer.

A handler that needs the gate's verdict on its own call (the Local PC
computer tool, which runs a disk-touching argument only once the user
approved it) reads it with :func:`approved_for` while it runs, the way a
registry tool reads ``GATE_APPROVED``; the model cannot set it.
"""

import json
import logging
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from pydantic import BaseModel, ConfigDict

from backend.copilot.model import ChatSession

from . import Decision, check_action, refusal_message
from .content import Image
from .reads import release_held_read, screen_read
from .subject import Subject

logger = logging.getLogger(__name__)

SubjectOf = Callable[[], Awaitable[Subject | None]]


class Verdict(BaseModel):
    """The gate's decision on a call that proceeds, and the subject it decided
    on, where it resolved one: an approval of exactly this call skips that."""

    model_config = ConfigDict(frozen=True)

    decision: Decision
    subject: Subject | None = None


_RUNNING: ContextVar[Verdict | None] = ContextVar(
    "non_registry_gate_verdict", default=None
)


async def gate_non_registry_tool(
    tool_name: str,
    args: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
    subject_of: SubjectOf | None = None,
) -> tuple[dict[str, Any] | None, Verdict | None]:
    """An MCP error payload to return instead of running, or None and the
    gate's verdict on a call that proceeds. ``subject_of`` resolves what the
    call acts on, as a registry tool's ``gate_subject`` does."""
    resolved: list[Subject | None] = []

    async def recorded() -> Subject | None:
        assert subject_of is not None
        subject = await subject_of()
        resolved.append(subject)
        return subject

    try:
        decision = await check_action(
            tool_name,
            args,
            user_id,
            session,
            subject_of=recorded if subject_of is not None else None,
        )
    except Exception:
        logger.warning(f"Action gate failed for MCP tool {tool_name}", exc_info=True)
        return (
            _error(
                tool_name,
                "This action could not be checked against your approval settings, "
                "so nothing ran.",
                None,
                session,
            ),
            None,
        )

    if decision.allowed:
        return None, Verdict(decision=decision, subject=next(iter(resolved), None))
    return _error(tool_name, decision.reason, decision.review_id, session), None


@contextmanager
def running_under(verdict: Verdict | None) -> Iterator[None]:
    """Run a non-registry handler under the gate's verdict on its call."""
    token = _RUNNING.set(verdict)
    try:
        yield
    finally:
        _RUNNING.reset(token)


def approved_for(subject_key: str) -> bool:
    """Whether the non-registry call now running proceeds on the user's word
    for ``subject_key``: their approval of exactly this call on a card, or a
    verdict the gate reached on that subject (their rule, or the chat's
    mode). False where the gate is off or cannot ask, and where it decided on
    another subject."""
    verdict = _RUNNING.get()
    if verdict is None or not verdict.decision.decided:
        return False
    if verdict.decision.approved:
        return True
    return verdict.subject is not None and verdict.subject.key == subject_key


async def release_non_registry_read(
    tool_name: str,
    args: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
) -> dict[str, Any] | None:
    """The held read's stored envelope or its stub, or None to run the read."""
    try:
        release = await release_held_read(tool_name, args, user_id, session)
    except Exception:
        logger.warning(f"Held-read lookup failed for {tool_name}", exc_info=True)
        return _error(
            tool_name,
            "This read could not be checked against your approvals, so nothing ran.",
            None,
            session,
        )
    if release is None:
        return None
    if release.released:
        return json.loads(release.output)
    return {"content": [{"type": "text", "text": release.output}], "isError": True}


async def screen_non_registry_read(
    tool_name: str,
    args: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
    result: dict[str, Any],
    *,
    outside: tuple[Any, ...] | None = None,
    full: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``result`` as capped for the model, or the stub that replaces it.

    ``outside`` is what the handler declared came from outside AutoGPT, placed
    in ``full``, its result before the cap; None judges all of ``result``."""
    text = _text_of(result)
    blocks = [b for b in result.get("content") or () if isinstance(b, dict)]
    images = tuple(
        Image(mime_type=str(b.get("mimeType", "")), data_base64=str(b["data"]))
        for b in blocks
        if b.get("type") == "image" and b.get("data")
    )
    stub = await screen_read(
        tool_name,
        args,
        user_id,
        session,
        output=json.dumps(result),
        success=not result.get("isError"),
        text=text,
        images=images,
        # The MCP handler never sees the SDK's tool_use_id; registry tools
        # on this engine use the same stand-in.
        tool_call_id=f"sdk-{uuid.uuid4().hex[:12]}",
        outside=outside,
        full=_text_of(full or result),
    )
    if stub is None:
        return result
    return {"content": [{"type": "text", "text": stub}], "isError": True}


def _text_of(result: dict[str, Any]) -> str:
    blocks = [b for b in result.get("content") or () if isinstance(b, dict)]
    return "\n".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")


def _error(
    tool_name: str,
    reason: str,
    review_id: str | None,
    session: ChatSession,
) -> dict[str, Any]:
    payload = {
        "type": "approval_required",
        "tool_name": tool_name,
        "reason": reason,
        "message": refusal_message(reason, review_id),
        "session_id": session.session_id,
    }
    if review_id:
        payload["review_id"] = review_id
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": True}
