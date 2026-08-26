"""Framing and chain policy shared by the two cross-expert handoff tools.

``delegate_to_expert`` and ``handoff_to_expert`` both open their prompt with a
``[...]`` preamble that interpolates the calling expert's name. Those names are
user-authored, so the sanitiser below is a safety invariant, not formatting —
and it lived in both files until a hardening landed in one and not the other,
leaving the twin forgeable. It lives here now so there is one place to fix.

The same is true of the chain bound: both tools write the same
``delegated_by_session_id`` provenance and both mint a session that runs a full
turn, so a bound that lives in only one of them is no bound at all — the model
just passes the task on with the other tool.
"""

import logging

from backend.api.features.experts.models import Expert
from backend.copilot.model import ChatSession, get_chat_session
from backend.data.db_accessors import experts_db

logger = logging.getLogger(__name__)

# Long enough for a real name, short enough that a crafted one cannot bury the
# preamble's own instructions under padding.
CALLER_NAME_LIMIT = 80

# How many hops a single task may travel between experts, whether delegated or
# handed over. Each hop is a fresh session with a fresh delegator, so without
# this nothing downstream would ever notice a chain — or a loop — sustaining
# itself on the user's credits.
MAX_DELEGATION_DEPTH = 3

# The preamble delimits itself with square brackets, so a name containing them
# can close the framing early and open a block of its own.
_FRAMING_DELIMITERS = str.maketrans("", "", "[]")


def safe_caller_name(caller: str) -> str:
    """Collapse *caller* to a single bracket-free line, capped and non-empty.

    Truncating after stripping matters: a name that is all brackets must not
    spend the budget and then collapse to nothing.
    """
    one_line = " ".join(caller.split()).translate(_FRAMING_DELIMITERS)
    return one_line[:CALLER_NAME_LIMIT].strip() or "a teammate"


async def resolve_target_expert(user_id: str, reference: str) -> Expert | None:
    """Resolve a teammate by id, falling back to a unique name match.

    The roster block that carries expert ids is injected only into the first
    user message, so a session opened before the team was hired knows its
    teammates by name alone. An id miss therefore retries as a
    case-insensitive name lookup; an ambiguous name stays unresolved so the
    model is told the roster instead of the tool guessing between twins.

    ``Expert.id`` is a plain string column, so a bare name simply misses here
    rather than raising — which leaves any exception meaning a real database
    failure. Those propagate to the caller, whose handler says "try again"
    instead of the flat lie that the teammate does not exist.
    """
    expert = await experts_db().get_expert(user_id, reference, include_workflows=False)
    if expert is not None:
        return expert
    wanted = reference.strip().casefold()
    if not wanted:
        return None
    matches = [
        e
        for e in await experts_db().list_experts(user_id, with_metrics=False)
        if not e.is_archived and e.name.strip().casefold() == wanted
    ]
    return matches[0] if len(matches) == 1 else None


async def unknown_target_message(
    user_id: str, reference: str, exclude_expert_id: str | None
) -> str:
    """A lookup-failure message that carries the roster the model is missing.

    Sessions older than the team never saw a ``<team_context>`` block, so a
    bare "no such expert" leaves the model with nothing to retry with.

    Only teammates who can actually take work are offered: a paused expert is
    refused by both delegation tools, so naming one here would just buy
    another failed call.
    """
    fallback = (
        f"No active expert matching {reference!r} on this team. Pick a "
        "teammate from your team context."
    )
    try:
        experts = await experts_db().list_experts(user_id, with_metrics=False)
    except Exception as e:
        logger.warning(f"Roster lookup for delegation error failed: {e}")
        return fallback
    teammates = [
        e
        for e in experts
        if not e.is_archived
        and e.id != exclude_expert_id
        and e.schedules_paused_at is None
    ]
    if not teammates:
        return fallback
    roster = "; ".join(f"{e.name} (expert_id: {e.id})" for e in teammates)
    return (
        f"No active expert matching {reference!r} on this team. "
        f"Use one of these expert_ids: {roster}."
    )


async def chain_refusal(
    user_id: str, session: ChatSession, target: Expert
) -> str | None:
    """Why *session* may not pass its task to *target*, or ``None`` if it may.

    Refuses a hop that would push the chain past its bound, and one that hands
    work to an expert already holding it further up the chain.

    Depth is read off the ``delegated_by_session_id`` provenance rather than a
    stored counter: a plain session pays nothing, and a delegated one pays at
    most ``MAX_DELEGATION_DEPTH`` cache-backed session reads. The ``seen`` set
    is belt-and-braces — provenance is written once at creation, but a
    traversal that trusts stored ids must not be able to spin.
    """
    seen = {session.session_id}
    parent_id = session.metadata.delegated_by_session_id
    depth = 0
    while parent_id and parent_id not in seen:
        depth += 1
        if depth >= MAX_DELEGATION_DEPTH:
            return (
                "This task has already been passed between teammates "
                f"{depth} times. Do as much as you can with it yourself "
                "instead of passing it on again."
            )
        seen.add(parent_id)
        parent = await get_chat_session(parent_id, user_id)
        if parent is None:
            break
        if parent.expert_id == target.id:
            return (
                f"{target.name} already has this task further up the chain, "
                "so passing it back would loop. Do what you can with it "
                "yourself instead."
            )
        parent_id = parent.metadata.delegated_by_session_id
    return None
