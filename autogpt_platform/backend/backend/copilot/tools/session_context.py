"""Build the per-session ``<session_context>`` block.

This block is injected by ``inject_user_context`` into the first user
message of every session so the model has *proactive* awareness of any
pending follow-ups it (or the user) previously scheduled — without
needing to call ``list_schedules`` on every greeting.

It carries:

* ``session_id`` — lets the model answer "cancel that" / "what did I
  schedule on this session" without guessing the id.
* ``pending_followups`` — count and a compact list (max 5; older ones
  collapsed into ``... +K more``) of the follow-ups currently queued
  against this session.

When there are zero pending follow-ups the block is rendered as a
single-line summary to save tokens — the model only needs the count
to know there is nothing to act on.

Like ``<env_context>``, ``<user_context>`` and ``<available_skills>``,
this block lands inside the **per-turn user message** (after the last
cache_control breakpoint) so injecting it does not bust the prefix
cache.  The system prompt itself is unchanged across sessions.
"""

import logging

from backend.executor.scheduler import CopilotTurnJobInfo
from backend.util.clients import get_scheduler_client

logger = logging.getLogger(__name__)

# Cap the per-block list — the model only needs a quick survey, not a
# full enumeration.  Anything beyond this is replaced with ``... +K more``
# to keep the prefix bounded for prompt-cache friendliness.
_MAX_LISTED_FOLLOWUPS = 5


def _format_one_followup(job: CopilotTurnJobInfo) -> str:
    """Render a single follow-up as one bullet line.

    Format examples::

        - "Check CI" (one-shot, fires 2026-05-22T13:50:00+00:00)
        - "Daily summary" (cron `0 9 * * *`, next fire 2026-05-23T09:00:00+00:00)

    The job's ``name`` is quoted so a name containing spaces or commas
    parses unambiguously when the model echoes it back to the user.
    Quotes inside the name are escaped to keep the surrounding quotes
    delimiting cleanly.
    """
    safe_name = (job.name or "follow-up").replace('"', '\\"')
    if job.cron:
        return f'- "{safe_name}" (cron `{job.cron}`, next fire {job.next_run_time})'
    return f'- "{safe_name}" (one-shot, fires {job.next_run_time})'


def _format_followup_list(jobs: list[CopilotTurnJobInfo]) -> list[str]:
    """Render up to ``_MAX_LISTED_FOLLOWUPS`` jobs and collapse the rest.

    Returns a list of pre-formatted lines (one per visible job, plus an
    optional ``... +K more`` line) so the caller can ``"\\n".join`` them
    into the block body without worrying about trailing newlines.
    """
    visible = jobs[:_MAX_LISTED_FOLLOWUPS]
    lines = [_format_one_followup(j) for j in visible]
    remaining = len(jobs) - len(visible)
    if remaining > 0:
        lines.append(f"... +{remaining} more")
    return lines


async def build_session_context(session_id: str, user_id: str) -> str:
    """Return the body of the ``<session_context>`` block for this turn.

    Calls the polymorphic ``get_execution_schedules`` endpoint with
    ``kind="copilot_turn"`` + ``session_id`` so only follow-ups bound
    to *this* session are returned (other-session followups for the
    same user are intentionally excluded — the model would have no
    handle to act on them mid-turn anyway).

    On any scheduler error the block degrades to the bare
    ``session_id`` line — the model still benefits from knowing the
    session it is in, and the turn never fails because of a transient
    scheduler RPC issue.

    The return value is the **body** of the block (no surrounding
    ``<session_context>`` tags); the caller wraps it.
    """
    try:
        raw_jobs = await get_scheduler_client().get_execution_schedules(
            user_id=user_id,
            session_id=session_id,
            kind="copilot_turn",
        )
    except Exception as e:
        # Graceful degradation: scheduler RPC issues must never fail the
        # turn — we still emit the session_id so the model knows which
        # session it is in.
        logger.warning(
            "build_session_context: scheduler RPC failed for session %s (%s); "
            "falling back to session_id-only block",
            session_id,
            e,
        )
        return f"session_id: {session_id}; pending_followups: 0"

    # The endpoint already narrows by ``kind`` server-side; the isinstance
    # filter is a belt-and-braces guard against a legacy untyped row that
    # might slip through (matches ``v1.list_copilot_turn_schedules``).
    jobs = [j for j in raw_jobs if isinstance(j, CopilotTurnJobInfo)]

    if not jobs:
        # Zero-follow-up sessions are the common case — collapse to one
        # line to keep the per-turn prefix small.
        return f"session_id: {session_id}; pending_followups: 0"

    lines = [
        f"session_id: {session_id}",
        f"pending_followups: {len(jobs)}",
    ]
    lines.extend(_format_followup_list(jobs))
    return "\n".join(lines)
