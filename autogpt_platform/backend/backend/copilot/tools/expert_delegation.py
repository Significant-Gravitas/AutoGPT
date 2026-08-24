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

from backend.api.features.experts.models import Expert
from backend.copilot.model import ChatSession, get_chat_session

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
