"""Whether untrusted content has entered a session, and what that escalates.

Taint is structural, not judged: it records that bytes we did not author
reached the model, never an opinion about whether they looked dangerous. An
attacker controls the phrasing of the content but not our control flow, so
this cannot be talked out of.

Two stores, because neither alone is correct:

* Redis holds the flag, written at gate time — *before* the taint source
  runs. Every MCP tool is annotated ``readOnlyHint=True`` so the CLI
  dispatches calls in parallel; writing after success would let a
  ``bash_exec`` issued in the same batch as a ``web_fetch`` read the flag
  before the fetch set it.
* The transcript is the durable backstop. A Redis flush, or a monthly
  ``schedule_followup`` firing into a session older than the key, must not
  hand back a clean bit while the injected text is still in the history the
  model resumes from.
"""

import logging

from backend.copilot.model import ChatSession
from backend.data.redis_client import get_redis_async

from .policy import TAINT_SOURCES

logger = logging.getLogger(__name__)

# Long enough that the transcript backstop is a genuine fallback rather than
# the usual path; short enough that dead sessions expire out of Redis.
_TAINT_TTL_SECONDS = 90 * 24 * 60 * 60
_TAINT_KEY = "copilot:gate:tainted:"
_ESCALATED_KEY = "copilot:gate:escalated:"


def _taint_key(session_id: str) -> str:
    return f"{_TAINT_KEY}{session_id}"


def _escalated_key(session_id: str, tool_name: str) -> str:
    """One key per (session, tool) rather than a set: the cluster client's set
    operations are not typed as awaitable, and a flag per tool is all this
    needs."""
    return f"{_ESCALATED_KEY}{session_id}:{tool_name}"


def born_tainted(session: ChatSession) -> bool:
    """Sessions whose *prompt* is untrusted, not just their tool output.

    ``platform_linking.chat`` creates chat-platform sessions without an
    explicit origin, so they default to ``interactive``, and ``_resolve_owner``
    maps server context to the server OWNER — any member of a linked Discord /
    Slack / Teams server drives a session billed to the linking user. The
    classifier is handed the turn's user message as the trusted premise for
    "is this what the user asked for", so here that premise is attacker-
    authored while no taint source is ever touched.
    """
    return session.metadata.source_platform is not None


def transcript_tainted(session: ChatSession) -> bool:
    """True if a taint source appears anywhere in the session history.

    Tool *result* rows carry only a ``tool_call_id``, so the names live on the
    assistant rows that requested them.
    """
    for message in session.messages:
        for call in message.tool_calls or ():
            name = (call.get("function") or {}).get("name")
            if isinstance(name, str) and name in TAINT_SOURCES:
                return True
    return False


async def is_tainted(session: ChatSession) -> bool:
    if born_tainted(session) or transcript_tainted(session):
        return True
    try:
        redis = await get_redis_async()
        return await redis.get(_taint_key(session.session_id)) is not None
    except Exception:
        # The transcript scan above already answered from durable state; a
        # Redis outage must not silently downgrade a session to clean, but it
        # also must not fail the tool call.
        logger.warning(
            f"Gate could not read taint for session {session.session_id}; "
            "assuming tainted",
            exc_info=True,
        )
        return True


async def mark_tainted(session_id: str, tool_name: str) -> None:
    """Record that *tool_name* is about to bring untrusted bytes in."""
    if tool_name not in TAINT_SOURCES:
        return
    try:
        redis = await get_redis_async()
        await redis.setex(_taint_key(session_id), _TAINT_TTL_SECONDS, tool_name)
    except Exception:
        logger.warning(
            f"Gate could not persist taint for session {session_id}", exc_info=True
        )


async def escalate(session_id: str, tool_name: str) -> None:
    """Force ASK for *tool_name* for the rest of this session.

    Monotone by construction — there is no de-escalate. Written when the user
    rejects a call, so re-proposing it with a space added to the command
    cannot buy a fresh verdict, and available as the seam a third-party veto
    (an authority review, a policy check) writes into without either side
    importing the other.
    """
    try:
        redis = await get_redis_async()
        await redis.setex(
            _escalated_key(session_id, tool_name), _TAINT_TTL_SECONDS, "1"
        )
    except Exception:
        logger.warning(
            f"Gate could not persist escalation for session {session_id}",
            exc_info=True,
        )


async def is_escalated(session_id: str, tool_name: str) -> bool:
    try:
        redis = await get_redis_async()
        return await redis.get(_escalated_key(session_id, tool_name)) is not None
    except Exception:
        logger.warning(
            f"Gate could not read escalations for session {session_id}; "
            "assuming escalated",
            exc_info=True,
        )
        return True
